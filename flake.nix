{
  description = "Python 3.13 development environment for ML diffusion project";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-25.11";
    gitignore = {
      url = "github:hercules-ci/gitignore.nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    torch-bin.url = "github:sirati/nix-torch-bin-pascal/feature-py-version-resolve";
    torch-bin.inputs.nixpkgs.follows = "nixpkgs";
  };

  outputs =
    {
      self,
      nixpkgs,
      gitignore,
      torch-bin,
    }:
    let
      systems = [
        "x86_64-linux"
        "aarch64-linux"
        "x86_64-darwin"
        "aarch64-darwin"
      ];
      forAllSystems = nixpkgs.lib.genAttrs systems;

      torchPackages = torch-bin.pytorch-packages;

      pkgsFor =
        system:
        import nixpkgs {
          inherit system;
          config = {
            allowUnfree = true;
          };
        };

      mlResultFor =
        system:
        let
          pkgs = pkgsFor system;
        in
        torchPackages.concretise {
          inherit pkgs;
          mlPackages = with torchPackages; [
            torch
            flash-attn
            mamba-ssm
          ];
          python = "3.13";
          cuda = "12.8";
          torch = "2.10";
          pascal = false;
          allowBuildingFromSource = true;
          extraPythonPackages =
            ps: with ps; [
              numpy
              pandas
              scipy
              scikit-learn
              tqdm
              matplotlib
              tensorboard
            ];
        };

      # wandb has torchvision as a nativeBuildInput (test dep). nixpkgs's source-built
      # torchvision tries to inherit cudaSupport from torch, which torch-bin's torch does
      # not carry. We use overrideScope inside extendEnv so that torchvision resolves to
      # torchvision-bin within the same concretise-augmented package set — meaning
      # torchvision-bin's torch dep resolves to torch 2.10 from concretise, not the
      # nixpkgs torch 2.9.1.
      withWandb =
        ps:
        let
          ps' = ps.overrideScope (_: prev: { torchvision = prev.torchvision-bin; });
        in
        [ ps'.wandb ];

      deploymentPackages =
        pkgs: with pkgs; [
          openssl
          cudaPackages.cudatoolkit
          cudaPackages.cudnn
        ];

      dockerOnlyPackages =
        pkgs: with pkgs; [
          bash
          coreutils
        ];

      devPackages =
        pkgs: with pkgs; [
          basedpyright
          nil
          nixd
          vscode-json-languageserver
          bash-language-server
          package-version-server
        ];
    in
    {
      devShells = forAllSystems (
        system:
        let
          pkgs = pkgsFor system;
          devEnv = (mlResultFor system).extendEnv (
            ps:
            let # todo this is dublicated with withWandb, lets merge that code path
              ps' = ps.overrideScope (_: prev: { torchvision = prev.torchvision-bin; });
            in
            with ps';
            [
              pip
              ruff
              wandb
            ]
          );
        in
        {
          default = pkgs.mkShell {
            packages = [ devEnv ] ++ (deploymentPackages pkgs) ++ (devPackages pkgs);

            shellHook = ''
              echo "╔════════════════════════════════════════════════════════════╗"
              echo "║  Python 3.13 ML development environment                    ║"
              echo "╚════════════════════════════════════════════════════════════╝"
              echo ""
              echo "Python version: $(python --version)"
              echo "CUDA Toolkit: ${pkgs.cudaPackages.cudatoolkit.version}"
              echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'N/A')"
              echo "PyTorch CUDA: $(python -c 'import torch; print(torch.version.cuda if hasattr(torch.version, "cuda") else "N/A")' 2>/dev/null || echo 'N/A')"
              echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())' 2>/dev/null || echo 'N/A')"
              echo ""
              echo "Packages installed:"
              echo "  ✓ PyTorch (torch-bin)"
              echo ""
              echo "Ready to train!"
              export bin_python=$(which python)
              export bin_python3=$(which python3)
              export CUDA_PATH="${pkgs.cudaPackages.cudatoolkit}"
              export LD_LIBRARY_PATH="${pkgs.cudaPackages.cudatoolkit}/lib:${pkgs.cudaPackages.cudnn}/lib:$LD_LIBRARY_PATH"
            '';
          };
        }
      );

      packages = forAllSystems (
        system:
        let
          pkgs = pkgsFor system;
          deploymentEnv = (mlResultFor system).extendEnv withWandb;
          inherit (gitignore.lib) gitignoreSource;

          projectSource = pkgs.lib.cleanSourceWith {
            src = gitignoreSource ./.;
            filter =
              path: type:
              let
                baseName = baseNameOf path;
              in
              !(pkgs.lib.hasPrefix "." baseName)
              && baseName != "flake.nix"
              && baseName != "flake-with-overlay.nix"
              && baseName != "flake.lock"
              && baseName != "result";
          };

          projectFiles = pkgs.runCommand "ml-diffusion-source" { } ''
            mkdir -p $out/app
            cp -r ${projectSource}/. $out/app/
            chmod -R +w $out/app
          '';
        in
        {
          dockerImage = pkgs.dockerTools.buildLayeredImage {
            name = "ml-diffusion";
            tag = "latest";

            contents = [
              deploymentEnv
              projectFiles
            ]
            ++ (deploymentPackages pkgs)
            ++ (dockerOnlyPackages pkgs);

            config = {
              Entrypoint = [
                "${deploymentEnv}/bin/python"
                "-m"
              ];
              WorkingDir = "/app";
            };
          };

          default = self.packages.${system}.dockerImage;
        }
      );
    };
}
