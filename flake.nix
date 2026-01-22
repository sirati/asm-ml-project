{
  description = "Python 3.13 development environment for ML diffusion project";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-25.11";
    gitignore = {
      url = "github:hercules-ci/gitignore.nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    torch-cu126.url = "github:sirati/nix-torch-bin-pascal";
  };

  outputs =
    {
      self,
      nixpkgs,
      gitignore,
      torch-cu126,
    }:
    let
      # Support multiple systems
      systems = [
        "x86_64-linux"
        "aarch64-linux"
        "x86_64-darwin"
        "aarch64-darwin"
      ];
      forAllSystems = nixpkgs.lib.genAttrs systems;

      # Enable CUDA support for packages that need it
      PASCAL = true;

      pkgsWithCuda =
        system:
        import nixpkgs {
          inherit system;
          config = {
            cudaSupport = true;
            allowUnfree = true; # CUDA packages are unfree
          };
        };

      # Package definitions
      deploymentPythonPackages =
        system: python-pkgs: with python-pkgs; [
          # Deep Learning frameworks
          (if PASCAL then torch-cu126.packages.${system}.torch-bin-cu126-pascal-py313-v210 else torch)
          torchvision

          # Mamba and attention mechanisms (require CUDA support)
          mamba-ssm
          flash-attn
          causal-conv1d

          # Core ML packages
          numpy
          pandas
          scipy
          scikit-learn

          # Utilities
          tqdm
          einops

          # Triton (for custom kernels)
          triton

          # Visualization and logging
          matplotlib
          tensorboard
          wandb
        ];

      devPythonPackages =
        python-pkgs: with python-pkgs; [
          pip
          ruff
        ];

      deploymentPackages =
        pkgs: with pkgs; [
          openssl

          # CUDA support (for NVIDIA GPUs)
          cudaPackages.cudatoolkit
          cudaPackages.cudnn

          # Build tools needed for pip packages (mamba-ssm, flash-attn)
          # gcc
          # cmake
          # ninja
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
          pkgs = pkgsWithCuda system;
        in
        {
          default = pkgs.mkShell {
            packages =
              with pkgs;
              [
                (python313.withPackages (
                  python-pkgs: (deploymentPythonPackages system python-pkgs) ++ (devPythonPackages python-pkgs)
                ))
              ]
              ++ (deploymentPackages pkgs)
              ++ (devPackages pkgs);

            shellHook = ''
              echo "╔════════════════════════════════════════════════════════════╗"
              echo "║  Python 3.13 ML development environment                    ║"
              echo "╚════════════════════════════════════════════════════════════╝"
              echo ""
              echo "Python version: $(python --version)"
              echo "CUDA: ${pkgs.cudaPackages.cudatoolkit.version}"
              echo ""
              echo "Packages installed:"
              echo "  ✓ PyTorch with CUDA"
              echo "  ✓ mamba-ssm"
              echo "  ✓ flash-attn"
              echo "  ✓ causal-conv1d"
              echo "  ✓ triton"
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
          pkgs = pkgsWithCuda system;
          python = pkgs.python313.withPackages (deploymentPythonPackages system);
          inherit (gitignore.lib) gitignoreSource;

          # Filter source files using gitignore, plus exclude dot files, flake files, and result
          projectSource = pkgs.lib.cleanSourceWith {
            src = gitignoreSource ./.;
            filter =
              path: type:
              let
                baseName = baseNameOf path;
              in
              # Exclude dot files and directories
              !(pkgs.lib.hasPrefix "." baseName)
              &&
                # Exclude flake files
                baseName != "flake.nix"
              && baseName != "flake.lock"
              &&
                # Exclude nix build results
                baseName != "result";
          };

          # Create a derivation that contains the project files
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
              python
              projectFiles
            ]
            ++ (deploymentPackages pkgs)
            ++ (dockerOnlyPackages pkgs);

            config = {
              Entrypoint = [
                "${python}/bin/python"
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
