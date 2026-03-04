# TODO
- [x] 排查 completeness 异常的问题 -> Completeness 计算方式不正确，先排除
- [x] 输出 geotif 格式的结果，将深度图恢复为 point cloud -> 参考 [Official implementation of Sat-MVSF](https://gpcv.whu.edu.cn/data/3d/mvs.html) 恢复出了基本结果
- [] 进行训练性能、速度排查和修正
    - Train 阶段 GPU 显存占用率变化大
    - Predict DSM 阶段模型 inference 缓慢
- [] 加入 feature volume 的融合机制
    - MVSTER 中基于 reference view feature volume 和 wraped source view feature volume 的 QKV 计算，对多个 wraped source view feature volume 进行注意力权重的 aggregation
- [] 排查虚拟容器下运行时因未指定 GDAL 库导致的错误，可能需要在脚本中修改