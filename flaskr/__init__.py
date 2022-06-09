import json
import os
from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
import numpy as np

def create_app(test_config=None):
    # create and configure the app
    app = Flask(__name__, instance_relative_config=True)
    CORS(app, support_credentials=True, origins='http://localhost')

    

    

    @app.route('/')
    def index():
        return 'Hello World!'

    @app.route('/test', methods=['POST'])
    @cross_origin(supports_credentials=True)
    def test():

        # get data from request
        algorithm = request.form['checkboxalgorithm']

        harga_val = request.form['harga']
        partner_val = request.form['partner']
        competitor_val = request.form['competitor']

        input_file = request.files['input_file']
        
        listAlgorithm = ['Logistic Regression', 'Decision Tree', 'Naive Bayes', 'Random Forest']

        # convert number to known algorithm name
        temp = algorithm.split(',')
        algorithm = []
        for i in range(len(temp)):
            algorithm.append(listAlgorithm[i - 1])

        
        # ==============file process====================
        input_file_name = os.path.basename(input_file.filename)
        file_extension = os.path.splitext(input_file_name)[1]
        input_file_name = input_file_name.split('.')[0]
        input_file_name = input_file_name + '_' + str(hash(input_file_name)) + file_extension

        # save to flaskr/sample_data folder
        input_file.save(os.path.join(app.instance_path, 'sample_data', input_file_name))

        # ===========end of file process=================



        if algorithm.__len__() <= 1:
            print(algorithm)

            result = {}

            if algorithm[0] == "Logistic Regression":
                from .algorithm.clf_logistic_regression import initialization
                values = np.array([[harga_val, partner_val, competitor_val]])
                result = initialization(values, input_file_name)
            if algorithm[0] == "Decision Tree":
                from .algorithm.clf_decision_tree import initialization
                values = np.array([[harga_val, partner_val, competitor_val]])
                result = initialization(values, input_file_name)
            if algorithm[0] == "Naive Bayes":
                from .algorithm.clf_naive_bayes import initialization
                values = np.array([[harga_val, partner_val, competitor_val]])
                result = initialization(values, input_file_name)
            if algorithm[0] == "Random Forest":
                from .algorithm.clf_random_forest import initialization
                values = np.array([[harga_val, partner_val, competitor_val]])
                result = initialization(values, input_file_name)
            
            return jsonify({
                    "length": algorithm.__len__(),
                    "status": "success",
                    "x_input": [harga_val, partner_val, competitor_val],
                    "algorithm_used": algorithm,
                    "results": result,
                })
        else:

            arrayResult = []

            for i in range(len(algorithm)):
                if algorithm[i] == "Logistic Regression":
                    from .algorithm.clf_logistic_regression import initialization
                    values = np.array([[harga_val, partner_val, competitor_val]])
                    result = initialization(values, input_file_name)
                    arrayResult.append(result)
                if algorithm[i] == "Decision Tree":
                    from .algorithm.clf_decision_tree import initialization
                    values = np.array([[harga_val, partner_val, competitor_val]])
                    result = initialization(values, input_file_name)
                    arrayResult.append(result)
                if algorithm[i] == "Naive Bayes":
                    from .algorithm.clf_naive_bayes import initialization                        
                    values = np.array([[harga_val, partner_val, competitor_val]])
                    result = initialization(values, input_file_name)
                    arrayResult.append(result)
                if algorithm[0] == "Random Forest":
                    from .algorithm.clf_random_forest import initialization
                    values = np.array([[harga_val, partner_val, competitor_val]])
                    result = initialization(values, input_file_name)
                    arrayResult.append(result)
                    

            return jsonify({
                    "length": algorithm.__len__(),
                    "status": "success",
                    "x_input": [harga_val, partner_val, competitor_val],
                    "algorithm_used": algorithm,
                    "results": arrayResult,
                })
            


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
                        from .algorithm.clf_logistic_regression import initialization
                        
                        values = np.array([[harga_val, partner_val, competitor_val]])
                        result = initialization(values, input_file_name)
                        jsonnya = {
                            "status": "success",
                            "test_file": input_file_name,
                            "output": result,
                        }
                        return jsonify(jsonnya)
                    if algorithm == '2':
                        from .algorithm.clf_naive_bayes import initialization
                        
                        values = np.array([[harga_val, partner_val, competitor_val]])
                        result = initialization(values, input_file_name)
                        jsonnya = {
                            "status": "success",
                            "test_file": input_file_name,
                            "output": result,
                        }
                        return jsonify(jsonnya)
                    if algorithm == '3':
                        from .algorithm.clf_decision_tree import initialization
                        
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
                                "message": "error"
                            }
                        )


    return app