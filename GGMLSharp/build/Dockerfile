FROM ubuntu:25.04 AS base

ARG GGML_COMMIT=master

RUN apt-get update && \
    apt-get install -y git cmake build-essential ccache wget tar

WORKDIR /src

RUN git clone --filter=blob:none --no-checkout https://github.com/ggml-org/ggml.git && \
    cd ggml && \
    git checkout ${GGML_COMMIT} && \
   sed -i '4598s/$/;/' src/ggml-quants.c # This is a quick patch needed for older GGML versions to get it to compile for windows

WORKDIR /src/ggml/build

FROM base AS win-base

ENV LLVM_MINGW_VERSION=20250717
ENV LLVM_MINGW_BASE=llvm-mingw-${LLVM_MINGW_VERSION}-ucrt-ubuntu-22.04-x86_64
ENV LLVM_MINGW_ARCHIVE=${LLVM_MINGW_BASE}.tar.xz
ENV LLVM_MINGW_PATH=/src/${LLVM_MINGW_BASE}

WORKDIR /src

RUN wget https://github.com/mstorsjo/llvm-mingw/releases/download/${LLVM_MINGW_VERSION}/${LLVM_MINGW_ARCHIVE} && \
    tar -xf ${LLVM_MINGW_ARCHIVE} 


WORKDIR /src/ggml/build

FROM win-base AS win-x64-builder

ENV TOOLCHAIN_FILE=/src/ggml/build/toolchain-windows-x64.cmake

RUN echo "set(CMAKE_SYSTEM_NAME Windows)" > ${TOOLCHAIN_FILE} && \
    echo "set(CMAKE_C_COMPILER ${LLVM_MINGW_PATH}/bin/x86_64-w64-mingw32-clang)" >> ${TOOLCHAIN_FILE} && \
    echo "set(CMAKE_CXX_COMPILER ${LLVM_MINGW_PATH}/bin/x86_64-w64-mingw32-clang++)" >> ${TOOLCHAIN_FILE} && \
    echo "set(CMAKE_FIND_ROOT_PATH ${LLVM_MINGW_PATH}/x86_64-w64-mingw32)" >> ${TOOLCHAIN_FILE} && \
    echo "set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)" >> ${TOOLCHAIN_FILE} && \
    echo "set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)" >> ${TOOLCHAIN_FILE} && \
    echo "set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)" >> ${TOOLCHAIN_FILE}

RUN cmake .. \
    -DCMAKE_TOOLCHAIN_FILE=${TOOLCHAIN_FILE} \
    -DGGML_BUILD_TESTS=OFF \
    -DBUILD_SHARED_LIBS=ON

RUN cmake --build . --config Release -j $(nproc)

FROM win-base AS win-arm64-builder

ENV TOOLCHAIN_FILE=/src/ggml/build/toolchain-windows-arm64.cmake

RUN echo "set(CMAKE_SYSTEM_NAME Windows)" > ${TOOLCHAIN_FILE} && \
    echo "set(CMAKE_C_COMPILER ${LLVM_MINGW_PATH}/bin/aarch64-w64-mingw32-clang)" >> ${TOOLCHAIN_FILE} && \
    echo "set(CMAKE_CXX_COMPILER ${LLVM_MINGW_PATH}/bin/aarch64-w64-mingw32-clang++)" >> ${TOOLCHAIN_FILE} && \
    echo "set(CMAKE_FIND_ROOT_PATH ${LLVM_MINGW_PATH}/aarch64-w64-mingw32)" >> ${TOOLCHAIN_FILE} && \
    echo "set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)" >> ${TOOLCHAIN_FILE} && \
    echo "set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)" >> ${TOOLCHAIN_FILE} && \
    echo "set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)" >> ${TOOLCHAIN_FILE}

RUN cmake .. \
    -DCMAKE_TOOLCHAIN_FILE=${TOOLCHAIN_FILE} \
    -DBUILD_SHARED_LIBS=ON \
    -DGGML_BUILD_TESTS=OFF

RUN cmake --build . --config Release -j $(nproc)

FROM base AS linux-x64-builder

RUN cmake .. \
    -DBUILD_SHARED_LIBS=ON \
    -DGGML_BUILD_TESTS=OFF

RUN cmake --build . --config Release -j $(nproc)

FROM base AS linux-arm64-builder

ENV TOOLCHAIN_FILE=/src/ggml/build/toolchain-linux-arm64.cmake

RUN apt-get update && \
    apt-get install -y g++-aarch64-linux-gnu

RUN echo "set(CMAKE_SYSTEM_NAME Linux)" > ${TOOLCHAIN_FILE} && \
    echo "set(CMAKE_SYSTEM_PROCESSOR aarch64)" >> ${TOOLCHAIN_FILE} && \
    echo "set(CMAKE_C_COMPILER aarch64-linux-gnu-gcc)" >> ${TOOLCHAIN_FILE} && \
    echo "set(CMAKE_CXX_COMPILER aarch64-linux-gnu-g++)" >> ${TOOLCHAIN_FILE}

RUN cmake .. \
    -DCMAKE_TOOLCHAIN_FILE=${TOOLCHAIN_FILE} \
    -DBUILD_SHARED_LIBS=ON \
    -DGGML_BUILD_TESTS=OFF

RUN cmake --build . --config Release -j $(nproc)

FROM base

WORKDIR /out

ENV BASE=/src/ggml/build

COPY --from=linux-x64-builder $BASE/src/*.so /out/runtimes/linux-x64/native/
COPY --from=linux-arm64-builder $BASE/src/*.so /out/runtimes/linux-arm64/native/
COPY --from=win-x64-builder $BASE/bin/*.dll /out/runtimes/win-x64/native/
COPY --from=win-arm64-builder $BASE/bin/*.dll /out/runtimes/win-arm64/native/