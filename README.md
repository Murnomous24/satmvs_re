# TODO
- [x] 排查 completeness 异常的问题 -> Completeness 计算方式不正确，先排除
- [x] 输出 geotif 格式的结果，将深度图恢复为 point cloud -> 参考 [Official implementation of Sat-MVSF](https://gpcv.whu.edu.cn/data/3d/mvs.html) 恢复出了基本结果
- [] 进行训练性能、速度排查和修正
    - Train 阶段 GPU 显存占用率变化大
    - Predict DSM 阶段模型 inference 缓慢
- [] 加入 feature volume 的融合机制