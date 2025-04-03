from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd

app = Flask(__name__)

# Carrega o modelo salvo (certifique-se de que o caminho está correto)
try:
    model = joblib.load("gradient_boosting_model.pkl")
    print("Modelo carregado com sucesso!")
except Exception as e:
    print(f"Erro ao carregar o modelo: {e}")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Obter dados JSON da requisição
        data = request.get_json(force=True)
        print(f"Dados recebidos: {data}")  # Exibir os dados recebidos

        # Criar DataFrame com valores padrão (evitar None)
        input_df = pd.DataFrame([{
            "highbp": int(data.get("highbp", 0)),
            "highchol": int(data.get("highchol", 0)),
            "cholcheck": int(data.get("cholcheck", 0)),
            "bmi": float(data.get("bmi", 0.0)),
            "smoker": int(data.get("smoker", 0)),
            "stroke": int(data.get("stroke", 0)),
            "heartdiseaseorattack": int(data.get("heartdiseaseorattack", 0)),
            "physactivity": int(data.get("physactivity", 0)),
            "fruits": int(data.get("fruits", 0)),
            "veggies": int(data.get("veggies", 0)),
            "hvyalcoholconsump": int(data.get("hvyalcoholconsump", 0)),
            "anyhealthcare": int(data.get("anyhealthcare", 0)),
            "nodocbccost": int(data.get("nodocbccost", 0)),
            "genhlth": int(data.get("genhlth", 0)),
            "menthlth": int(data.get("menthlth", 0)),
            "physhlth": int(data.get("physhlth", 0)),
            "diffwalk": int(data.get("diffwalk", 0)),
            "sex": int(data.get("sex", 0)),
            "age": int(data.get("age", 0)),
            "education": int(data.get("education", 0)),
            "income": float(data.get("income", 0.0))
        }])

        print(f"DataFrame criado: {input_df}")  # Verificar o DataFrame antes da previsão

        # Fazer a previsão
        prediction = model.predict(input_df)
        result = "Diabetes" if prediction[0] == 1 else "Não Diabetes"

        # Retorna a resposta em formato JSON
        return jsonify({"prediction": int(prediction[0]), "result": result})

    except Exception as e:
        print(f"Erro ao fazer a predição: {e}")
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
