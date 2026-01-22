
The project is split into different modules to allow me to cleanly separate concerns and allow easier configuration and hyperparameter search.

1. Data Loading and tensorization logic
2. optional ML preprocessing layer
3. Overall Model architecture (transformer vs perceiver-io like)
   1. Encoder part
   2. Decoder part
4. Active training objectives
   - they each use a separate instance of the Decoder part
5.
