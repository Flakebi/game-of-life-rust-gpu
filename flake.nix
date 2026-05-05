{
  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
  };

  outputs =
    { self, nixpkgs }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs {
        inherit system;
      };

      ldLibraryPathHook = pkgs.writeText "ld-library-path-hook.sh" ''
        addLdLibraryPath () {
            if [ -z ''${LD_LIBRARY_PATH+x} ]; then
                export LD_LIBRARY_PATH=
            fi
            if [ -d "$1/lib" ]; then
                export LD_LIBRARY_PATH="$1/lib''${LD_LIBRARY_PATH:+:}$LD_LIBRARY_PATH"
            fi
        }
        runBinary() {
            LD_LIBRARY_PATH=$LD_LIBRARY_PATH $(< "$NIX_CC/nix-support/dynamic-linker") "$@"
        }
        addEnvHooks "$targetOffset" addLdLibraryPath
      '';
    in
    {
      devShells.${system}.default = pkgs.mkShell {
        packages = with pkgs; [
          ldLibraryPathHook
          llvmPackages_latest.clang-unwrapped
          rocmPackages.rocm-runtime
          rocmPackages.clr
          vulkan-loader
          wayland
          libxkbcommon

          # libstdc++
          stdenv.cc.cc.lib
        ];

        ROCM_PATH = "${pkgs.rocmPackages.clr}";
        ROCM_DEVICE_LIB_PATH = "${pkgs.rocmPackages.rocm-device-libs}";
      };
    };
}

