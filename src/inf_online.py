from flask import Flask, jsonify, request
from flask_cors import CORS
import copy
import os
import sys
from pathlib import Path
from img_blur_processor import IMGBlurProcessor

sys.path.insert(0, "./Mask2Former/demo")


# current_dir = os.path.dirname(os.path.abspath(__file__))
# sys.path.append(str(Path(current_dir).parent))
# sys.path.append(current_dir)


# app = Flask(__name__)
# CORS(app)


# @app.route('/backend/algorithm/local/playground', methods=['POST', 'GET'])
def playground():
    # input_data = request.get_json()
    # message = input_data['message']
    # output_data = copy.deepcopy(input_data)
    
    # img_blur_processor = IMGBlurProcessor([message['image']], 0.3, 5, "/share/songyuhao/generation/data/genlane_test", "Gaussian", True)
    img_blur_processor = IMGBlurProcessor(["/mnt/ve_share/songyuhao/generation/data/test/exp169/aaa.jpg"], 0.3, 5, "/mnt/ve_share/songyuhao/generation/data/genlan_test", "Gaussian", False)
    img_blur_processor.run()
    
    
if __name__ == '__main__':
    # app.run(host='0.0.0.0', port=80)
    playground()