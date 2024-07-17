from flask import Flask, render_template, request
import random
from hypernetx.classes.hypergraph import Hypergraph
from hypernetx.algorithms.matching_algorithms import greedy_matching, iterated_sampling, HEDCS_matching

app = Flask(__name__)

def parse_hypergraph(data):
    try:
        edges = {}
        for line in data.split('\n'):
            if line.strip():
                key, values = line.split(':')
                edges[key.strip()] = list(map(int, values.split(',')))
        return Hypergraph(edges)
    except Exception as e:
        raise ValueError(
            "Input data is not in the correct format. Each line should be in the format 'edge: vertex1,vertex2,...'")


@app.route('/', methods=['GET', 'POST'])
def index():
    error = None
    form_data = {'hypergraph_data': '', 'k': '', 's': '', 'algorithm': ''}
    if request.method == 'POST':
        try:
            hypergraph_data = request.form['hypergraph_data']
            k = int(request.form['k'])
            s = int(request.form['s'])
            algorithm = request.form['algorithm']

            hypergraph = parse_hypergraph(hypergraph_data)

            if algorithm == 'greedy':
                result = greedy_matching(hypergraph, k)
            elif algorithm == 'iterated':
                result = iterated_sampling(hypergraph, s)
            elif algorithm == 'hedcs':
                result = HEDCS_matching(hypergraph, s)
            else:
                result = "Invalid algorithm selected."

            return render_template('result.html', data=hypergraph_data, result=result)
        except ValueError as e:
            error = str(e)
        except Exception as e:
            error = "An error occurred. Please ensure all inputs are correct."

        form_data = request.form

    return render_template('index.html', error=error, form_data=form_data)


if __name__ == '__main__':
    app.run(debug=True)
