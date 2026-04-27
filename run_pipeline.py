import os
import kfp
from kfp_pipeline import stock_training_pipeline

def compile_and_submit():
    # 1. Compile pipeline
    package_path = 'pipeline.yaml'
    kfp.compiler.Compiler().compile(
        pipeline_func=stock_training_pipeline,
        package_path=package_path
    )
    print(f"Pipeline compiled to {package_path}")
    
    # 2. Lấy KFP host từ env
    kfp_host = os.environ.get("KFP_HOST", "http://ml-pipeline.kubeflow.svc.cluster.local:8888")
    
    try:
        # 3. Tạo client kết nối lên KFP 
        client = kfp.Client(host=kfp_host)
        
        # 4. Submit pipeline run
        experiment_name = 'Stock Prediction Automations'
        import time
        run_name = f'ARC_Triggered_Run_{int(time.time())}'
        
        # Lấy experiment (tạo mới nếu chưa có)
        experiment = client.create_experiment(name=experiment_name, namespace='kubeflow')
        
        # Submit execution 
        run_result = client.run_pipeline(
            experiment_id=experiment.experiment_id,
            job_name=run_name,
            pipeline_package_path=package_path,
            params={} 
        )
        print(f"Bắt đầu run thành công! ID: {run_result.run_id}")
        
    except Exception as e:
        print(f"Không thể kết nối đến KFP hoặc submit không thành công: {str(e)}")
        print("Xin hãy kiểm tra biến môi trường KFP_HOST")

if __name__ == '__main__':
    compile_and_submit()
