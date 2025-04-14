import numpy as np

min = {'Pregnancies': np.float64(0),
 'Glucose': np.float64(0),
 'BloodPressure': np.float64(0),
 'SkinThickness': np.float64(0),
 'Insulin': np.float64(0),
 'BMI': np.float64(0.0),
 'DiabetesPedigreeFunction': np.float64(0.078),
 'Age': np.float64(21),}

max = {'Pregnancies': np.float64(17),
 'Glucose': np.float64(199),
 'BloodPressure': np.float64(122),
 'SkinThickness': np.float64(99),
 'Insulin': np.float64(846),
 'BMI': np.float64(67.1),
 'DiabetesPedigreeFunction': np.float64(2.42),
 'Age': np.float64(81),}

# {}.k
def normalize(inp):
    print(inp.keys())
    newinp = {}

    
    for key in max.keys():
        # key = 
        newinp[key] = np.float64(inp[key])
        newinp[key] = (max[key] - newinp[key])/(max[key] - min[key])

    print(newinp)
    return newinp