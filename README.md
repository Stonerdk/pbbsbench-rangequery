# pbbsbench - RangeQuery in CUDA

```
git clone https://github.com/stonerdk/pbbsbench-rangequery
cd pbbsbench-rangequery
git submodule update

cd testData/geometryData/data
make 2DinCube_<데이터 수>

cd ../../../benchmarks/rangeQuery2d/gpu
nvcc -std=c++17 -I . range.cu -o range
./range ../../../testData/geometryData/data/2DinCube_<데이터 수>
```