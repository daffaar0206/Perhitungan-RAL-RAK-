from flask import Blueprint, render_template, request, jsonify
from app.utils import process_ral, process_rak, process_ral_factorial

bp = Blueprint('main', __name__)

@bp.route('/')
def index():
    return render_template('index.html')

@bp.route('/process', methods=['POST'])
def process():
    try:
        data = request.get_json()
        design_type = data.get('design_type', 'ral')
        
        if design_type == 'ral':
            result = process_ral(data['data'])
        elif design_type == 'rak':
            result = process_rak(data['data'])
        elif design_type == 'ral_factorial':
            result = process_ral_factorial(data['data'])
        else:
            return jsonify({'error': 'Tipe rancangan tidak valid'})
            
        return jsonify(result)
        
    except Exception as e:
        import traceback
        error_msg = str(e)
        stack_trace = traceback.format_exc()
        return jsonify({'error': f"{error_msg}\n{stack_trace}"})

@bp.route('/post_hoc', methods=['POST'])
def post_hoc():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data received'}), 400
            
        if 'data' not in data:
            return jsonify({'error': 'Missing required field: data'}), 400
            
        design_type = data.get('design_type', 'ral')
        post_hoc_type = data.get('post_hoc_type')
        
        if not post_hoc_type:
            return jsonify({'error': 'Missing required field: post_hoc_type'}), 400
        
        if design_type == 'ral':
            result = process_ral(data['data'], post_hoc_type)
        elif design_type == 'rak':
            result = process_rak(data['data'], post_hoc_type)
        elif design_type == 'ral_factorial':
            result = process_ral_factorial(data['data'], post_hoc_type)
        else:
            return jsonify({'error': 'Tipe rancangan tidak valid'}), 400
            
        return jsonify(result)
        
    except Exception as e:
        import traceback
        error_msg = str(e)
        stack_trace = traceback.format_exc()
        return jsonify({'error': f"{error_msg}\n{stack_trace}"}), 500
