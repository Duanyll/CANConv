pip3 install --user -r requirements.txt
git submodule update --init --recursive
cmake -DCMAKE_BUILD_TYPE:STRING=Release -Bbuild -G Ninja
cmake --build build --config Release --target all