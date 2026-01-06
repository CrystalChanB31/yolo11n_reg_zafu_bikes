from ultralytics import YOLO

def main():
    # Load a model
    model = YOLO("yolo11n.pt")  # load a pretrained model (recommended for training)

    # Train the model
    # data: path to our custom yaml
    # epochs: 50 is a good starting point, can be less for quick test
    # imgsz: 640 standard
    # device: 0 (GPU) if available
    results = model.train(
        data="datasets/unified/data.yaml", 
        epochs=200, 
        imgsz=640, 
        plots=True,
        batch= 64,
        name="yolo11n_nfon_motorized_vehicles",
        
        # 增强数据增强以改善类别不平衡
        hsv_h=0.015,      # 色调增强
        hsv_s=0.7,        # 饱和度增强  
        hsv_v=0.4,        # 明度增强
        degrees=10,       # 旋转角度（±10度）
        translate=0.1,    # 平移（10%）
        scale=0.5,        # 缩放（50%-150%）
        shear=2,          # 剪切（±2度）
        perspective=0.0,  # 透视变换
        flipud=0.0,       # 上下翻转（关闭，因为车辆不应该上下翻转）
        fliplr=0.5,       # 左右翻转（50%概率）
        mosaic=1.0,       # 马赛克增强（100%概率）
        mixup=0.1,        # 混合增强（10%概率）
        
        # 优化器参数
        optimizer='AdamW',
        lr0=0.001,        # 初始学习率
        lrf=0.01,         # 最终学习率因子
        momentum=0.937,
        weight_decay=0.0005,
        
        # 损失函数权重（如果支持）
        # cls=0.5,  # 分类损失权重（可以稍微降低，让模型更关注定位）
    )

if __name__ == "__main__":
    main()
