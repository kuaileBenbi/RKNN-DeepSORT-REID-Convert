from rknn.api import RKNN

MODEL_PATH = 'deepsort.onnx'
RKNN_MODEL = 'deepsort.rknn'

rknn = RKNN(verbose=True)

do_quant = False

# Pre-process config
print('--> Config model')
rknn.config(mean_values=[[0, 0, 0]], std_values=[[255, 255, 255]], target_platform='rk3588')
print('done')

# Load model
print('--> Loading model')
ret = rknn.load_onnx(MODEL_PATH)

if ret != 0:
    print('Load model failed!')
    exit(ret)
print('done')

# Build model
print('--> Building model')
ret = rknn.build(do_quantization=False)
if ret != 0:
    print('Build model failed!')
    exit(ret)
print('done')


# Export rknn model
print('--> Export rknn model')
ret = rknn.export_rknn(RKNN_MODEL)
if ret != 0:
    print('Export rknn model failed!')
    exit(ret)
print('done')

# Release
rknn.release()




