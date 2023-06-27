from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig
from flask import Flask, jsonify, request
from flask_cors import CORS
from flaskext.mysql import MySQL #pip install flask-mysql
import pymysql
model_path = "./bloomz-lora-vi-chatbot/"
config = PeftConfig.from_pretrained(model_path)

# load base LLM model and tokenizer
tokenizer_from_file = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
model_from_file = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path).to("cuda")


# configuration
DEBUG = True
 
# instantiate the app
app = Flask(__name__)
app.config.from_object(__name__)
 
# enable CORS
CORS(app, resources={r'/*': {'origins': '*'}})

@app.route('/search', methods=['GET'])
def receive_string():
    string = request.args

    string_get = string.get("message")
    # do something with the string
    print(string.get("message"))
    
    question3 = string_get
    prompt = "Hỏi:"+ question3 + '''
        Đáp:'''
    prompt
    # In ra kết quả
    inputs = tokenizer_from_file( prompt, return_tensors="pt")
    # Do biến inputs được lưu trên cpu mà model thì load trên gpu nên phải chuyển
    # biến này lên gpu bằng hàm .to("cuda")
    data = {}
    with torch.autocast("cuda"):
        outputs = model_from_file.generate(input_ids=inputs["input_ids"].to("cuda"),
                                max_new_tokens=100,
                                no_repeat_ngram_size=3,
                                num_beams=3,
                                num_return_sequences=3)
        count = 1
        for i in tokenizer_from_file.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True):
            print(i)
            data[count] = i
            count += 1
    
    return jsonify(data)
