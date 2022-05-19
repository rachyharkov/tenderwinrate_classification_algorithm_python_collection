import os
from flask import Flask, request, jsonify
import numpy as np

def create_app(test_config=None):
    # create and configure the app
    app = Flask(__name__, instance_relative_config=True)

    @app.route('/')
    def index():
        return 'Hello World!'

    @app.route('/predict', methods=['POST'])
    def run_predict():
        # check if there is no data sent
        if not request.form:
            return jsonify({'error': 'No data received'})

        algorithm = request.form['algorithm']
        
        if not algorithm:
            return jsonify(
                {
                    "status": "error",
                    "message": "Please select algorithm: [1. Random Forest] [2. Decision Tree] [3. Logistic Regression]"
                }
            )
        else:
            harga_val = request.form['harga']
            partner_val = request.form['partner']
            competitor_val = request.form['competitor']

            input_file = request.files['input_file']

            # check if the input is valid
            if not harga_val or not partner_val or not competitor_val:
                return jsonify(
                    {
                        "status": "error",
                        "message": "We are detected your algorithm choice, we need harga, partner and competitor value on body"
                    }
                )
            else:

                # upload input_file
                if input_file:
                    
                    # change file name to md5 hash
                    input_file_name = os.path.basename(input_file.filename)
                    file_extension = os.path.splitext(input_file_name)[1]
                    input_file_name = input_file_name.split('.')[0]
                    input_file_name = input_file_name + '_' + str(hash(input_file_name)) + file_extension

                    # save to flaskr/sample_data folder
                    input_file.save(os.path.join(app.instance_path, 'sample_data', input_file_name))
                    
                    if algorithm == '1':
                        # run random forest\
                        from .algorithm.clf_random_forest import initialization
                        
                        values = np.array([[harga_val, partner_val, competitor_val]])
                        result = initialization(values, input_file_name)
                        jsonnya = {
                            "status": "success",
                            "test_file": input_file_name,
                            "output": result,
                        }
                        return jsonify(jsonnya)
                    if algorithm == '2':
                        # run random forest\
                        from .algorithm.clf_decision_tree import initialization
                        
                        values = np.array([[harga_val, partner_val, competitor_val]])
                        result = initialization(values, input_file_name)
                        jsonnya = {
                            "status": "success",
                            "test_file": input_file_name,
                            "output": result,
                        }
                        return jsonify(jsonnya)
                    if algorithm == '3':
                        # run random forest\
                        from .algorithm.clf_logistic_regression import initialization
                        
                        values = np.array([[harga_val, partner_val, competitor_val]])
                        result = initialization(values, input_file_name)
                        jsonnya = {
                            "status": "success",
                            "test_file": input_file_name,
                            "output": result,
                        }
                        return jsonify(jsonnya)
                    else:
                        return jsonify(
                            {
                                "status": "success",
                                "message": "anu"
                            }
                        )


    return app