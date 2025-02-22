{
  description = "A flake for python development environment, using nix for python package management.";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
		coosis-nix.url = "github:Coosis/coosis-nix";
  };

  outputs = { self, nixpkgs, coosis-nix, ... }:
    let
      # to work with older version of flakes
      lastModifiedDate = self.lastModifiedDate or self.lastModified or "19700101";

      # Generate a user-friendly version number.
      version = builtins.substring 0 8 lastModifiedDate;

      # System types to support.
      supportedSystems = [ "x86_64-linux" "x86_64-darwin" "aarch64-linux" "aarch64-darwin" ];

      # Helper function to generate an attrset '{ x86_64-linux = f "x86_64-linux"; ... }'.
      forAllSystems = nixpkgs.lib.genAttrs supportedSystems;

      # Nixpkgs instantiated for supported system types.
      nixpkgsFor = forAllSystems (system: import nixpkgs { inherit system; });
    in
    {
      # Add dependencies that are only needed for development
      devShells = forAllSystems (system:
        let
          pkgs = nixpkgsFor.${system};
          # Choose desired Python version
          # python39 = pkgs.python39;
          # python310 = pkgs.python310;
          # python311 = pkgs.python311;
          python = pkgs.python3;
          # python313 = pkgs.python313;
					cpkgs = coosis-nix.packages.${system};
        in
        {
          default = pkgs.mkShell {
            buildInputs = with pkgs; [
              (python.withPackages (python-pkgs: with python-pkgs;
              [
                # A Python interpreter including the 'venv' module is required to bootstrap
                # the environment.
                python
                pip
								virtualenv

                # Desired Python packages
                numpy
								torch
								pandas
								scikit-learn
								antlr4-python3-runtime
								flask
                # If nixpkgs don't have it or you want to use a different version,
                # you can fetch it from PyPI
                # For wheels, check out fetchPypi src: 
                # https://github.com/NixOS/nixpkgs/blob/master/pkgs/build-support/fetchpypi/default.nix
                (buildPythonPackage rec {
                  pname = "py-solc-x";
                  version = "2.0.3";
                  src = pkgs.fetchPypi {
                    inherit pname version;
                    sha256 = "sha256-G1/ibV9JdrH8wrMnb7L+GCYzvDV1ejq/O5r1rFfsJUQ=";
                  };

									buildInputs = [ pip ];

                  meta = with pkgs.lib; {
                    description = "py-solc-x is a Python wrapper around the solc Solidity compiler.";
                    license = licenses.mit;
                    homepage = "https://github.com/ApeWorX/py-solc-x";
                  };
                })
              ]))

              # Other packages you might need in your environment
              # ...
							# solc
							# solc-select
							slither-analyzer
							cpkgs.solidity-language-server
							graphviz
							antlr
							jdk
							fd
							curl
							unzip
            ];
						venvDir = "./venv";
						shellHook = ''
							export VENV_DIR="./venv"
							if [ ! -d "$VENV_DIR" ]; then
								echo "Creating Python virtual environment in $VENV_DIR"
								python -m venv $VENV_DIR
							fi
							source $VENV_DIR/bin/activate
							export SOLCX_BINARY_PATH=./.solcx/
							export SHELL=$(which zsh)
							exec zsh
						'';
          };
        });
    };
}
