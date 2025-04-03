import streamlit as st
import joblib
import numpy as np

# Carregar o modelo treinado
modelo = joblib.load("gradient_boosting_model.pkl")


st.title("Predição de Diabetes")
st.write("Preencha os dados abaixo para verificar a probabilidade de ter diabetes.")

# Criar inputs no Streamlit
highbp = st.checkbox("Possui pressão alta?")
highchol = st.checkbox("Possui alto colesterol?")
cholcheck = st.checkbox("Fez a checagem do seu colesterol nos últimos cinco anos?")
bmi = st.slider("Índice de Massa Corporal (IMC)", 0.0, 100.0, 25.0)
smoker = st.checkbox("Já fumou pelo menos mais de 100 cigarros?")
stroke = st.checkbox("Possui algum problema de saúde crônico?")
heartdiseaseorattack = st.checkbox("Possui algum problema cardíaco ou já teve ataque cardíaco?")
physactivity = st.checkbox("Faz atividade física regularmente?")
fruits = st.checkbox("Consome pelo menos uma fruta ou mais por dia?")
veggies = st.checkbox("Consome pelo menos algum único tipo de vegetal ou mais por dia?")
hvyalcoholconsump = st.checkbox("Consome álcool 3 vezes por semana ou mais?")

# Dropdowns
genhlth = st.selectbox("Status de Saúde", ["Excelente", "Muito Bom", "Bom", "Médio", "Ruim"], index=2)
sex = st.selectbox("Sexo", ["Masculino", "Feminino"], index=0)

# Slider para idade
age = st.slider("Idade", 0, 100, 30)


# Mapeamento das respostas para os valores numéricos
def converter_checkbox(valor):
    return 1 if valor else 0


genhlth_mapping = {"Excelente": 5, "Muito Bom": 4, "Bom": 3, "Médio": 2, "Ruim": 1}
sex_mapping = {"Masculino": 1, "Feminino": 0}

# Criar array de entrada para o modelo
input_data = np.array([
    int(highbp), float(highchol), int(cholcheck), float(bmi), float(smoker), float(stroke),
    float(heartdiseaseorattack), int(physactivity), int(fruits), int(veggies),
    int(hvyalcoholconsump), genhlth_mapping[genhlth], int(sex_mapping[sex]), float(age)
]).reshape(1, -1)

# Botão para fazer a previsão
if st.button("Verificar probabilidade de diabetes"):
    resultado = modelo.predict(input_data)[0]
    probabilidade = modelo.predict_proba(input_data)[0][1] * 100

    if resultado == 1:
        st.error(f"Alerta: Há uma alta chance de diabetes ({probabilidade:.2f}%). Consulte um médico.")
    else:
        st.success(f"Boa notícia! A chance de diabetes é baixa ({probabilidade:.2f}%). Continue com hábitos saudáveis.")
