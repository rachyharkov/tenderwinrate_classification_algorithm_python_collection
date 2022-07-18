from ast import Return
import os
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS, cross_origin
import numpy as np

def convertAlgorithmSelection(data):
    # must be correctly positioned
    listAlgorithm = ['Logistic Regression', 'Decision Tree', 'Naive Bayes', 'Random Forest']

    temp = data.split(',')
    algorithm = []
    for i in range(len(temp)):
        algorithm.append(listAlgorithm[int(temp[i])])
    
    return algorithm

def saveDataset(file, app):
    input_file_name = os.path.basename(file.filename)
    file_extension = os.path.splitext(input_file_name)[1]
    input_file_name = input_file_name.split('.')[0]
    input_file_name = input_file_name + '_' + str(hash(input_file_name)) + file_extension

    # save to flaskr/sample_data folder
    file.save(os.path.join(app.instance_path, 'sample_data', input_file_name))
    # if file save success return true, else false
    print('file saved:' + input_file_name)
    return input_file_name

def create_app():
    # create and configure the app
    app = Flask(__name__, instance_relative_config=True)
    CORS(app, support_credentials=True, origins='http://localhost')

    @app.route('/')
    def index():
        return 'Hello World!'

    @app.route('/graph/', methods=['GET'])
    def graph():
        # get first parameter
        param1 = request.args.get('name')
        
        # return send images from graph_data folder
        return send_file(os.path.join(app.instance_path, 'graph_data', param1), mimetype='image/png')


    @app.route('/predict', methods=['POST'])
    @cross_origin(supports_credentials=True)
    def run_this_func():

        # get data from request
        selectedAlgorithm = request.form['checkboxalgorithm']

        harga_val = request.form['harga']
        partner_val = request.form['partner']
        competitor_val = request.form['competitor']

        input_file = request.files['input_file']
        
        # # convert number to known algorithm name
        convertedSelectedAlgorithm = convertAlgorithmSelection(selectedAlgorithm)
        
        #save dataset
        input_file_name = saveDataset(input_file, app)

        if convertedSelectedAlgorithm.__len__() <= 1:
            result = {}
            if convertedSelectedAlgorithm[0] == "Logistic Regression":
                from .algorithm.clf_logistic_regression import initialization
                values = np.array([[harga_val, partner_val, competitor_val]])
                result = initialization(values, input_file_name)
            if convertedSelectedAlgorithm[0] == "Decision Tree":
                from .algorithm.clf_decision_tree import initialization
                values = np.array([[harga_val, partner_val, competitor_val]])
                result = initialization(values, input_file_name)
            if convertedSelectedAlgorithm[0] == "Naive Bayes":
                from .algorithm.clf_naive_bayes import initialization
                values = np.array([[harga_val, partner_val, competitor_val]])
                result = initialization(values, input_file_name)
            if convertedSelectedAlgorithm[0] == "Random Forest":
                from .algorithm.clf_random_forest import initialization
                values = np.array([[harga_val, partner_val, competitor_val]])
                result = initialization(values, input_file_name)
            
            # remove file
            os.remove(os.path.join(app.instance_path, 'sample_data', input_file_name))

            return jsonify({
                    "length": convertedSelectedAlgorithm.__len__(),
                    "status": "success",
                    "x_input": [harga_val, partner_val, competitor_val],
                    "algorithm_used": convertedSelectedAlgorithm,
                    "results": result,
                })
        else:

            arrayResult = []

            for i in range(len(convertedSelectedAlgorithm)):
                if convertedSelectedAlgorithm[i] == "Logistic Regression":
                    from .algorithm.clf_logistic_regression import initialization
                    values = np.array([[harga_val, partner_val, competitor_val]])
                    result = initialization(values, input_file_name)
                    arrayResult.append(result)
                if convertedSelectedAlgorithm[i] == "Decision Tree":
                    from .algorithm.clf_decision_tree import initialization
                    values = np.array([[harga_val, partner_val, competitor_val]])
                    result = initialization(values, input_file_name)
                    arrayResult.append(result)
                if convertedSelectedAlgorithm[i] == "Naive Bayes":
                    from .algorithm.clf_naive_bayes import initialization                        
                    values = np.array([[harga_val, partner_val, competitor_val]])
                    result = initialization(values, input_file_name)
                    arrayResult.append(result)
                if convertedSelectedAlgorithm[i] == "Random Forest":
                    from .algorithm.clf_random_forest import initialization
                    values = np.array([[harga_val, partner_val, competitor_val]])
                    result = initialization(values, input_file_name)
                    arrayResult.append(result)
                    
            os.remove(os.path.join(app.instance_path, 'sample_data', input_file_name))
            return jsonify({
                    "length": convertedSelectedAlgorithm.__len__(),
                    "status": "success",
                    "x_input": [harga_val, partner_val, competitor_val],
                    "algorithm_used": convertedSelectedAlgorithm,
                    "results": arrayResult,
                })

    return app