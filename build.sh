dir=`pwd`

cmake --no-warn-unused-cli -DCMAKE_EXPORT_COMPILE_COMMANDS:BOOL=TRUE -DCMAKE_BUILD_TYPE:STRING=Release -Scanconv/layers/pwac -Bbuild -G Ninja
cmake --build build --config Release --target all --

git submodule update
cd kmcuda
cmake --no-warn-unused-cli -DCMAKE_EXPORT_COMPILE_COMMANDS:BOOL=TRUE -DCMAKE_BUILD_TYPE:STRING=Release -Ssrc -Bbuild -G Ninja
cmake --build build --config Release --target all --
cp build/libKMCUDA.so $dir/canconv/layers/kmeans/libKMCUDA.so