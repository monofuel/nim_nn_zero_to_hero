{ pkgs ? import <nixpkgs> {} }:

with pkgs;

mkShell {
  buildInputs = [
    python312     # Specify the Python version you need
    (python312Packages.numpy)
    (python312Packages.pytorch)
    # (python312Packages.torchvision)
    (jupyter)
    (python312Packages.matplotlib)  # Optional, for plotting
  ];
}