import streamlit as st, pandas as pd, joblib, matplotlib.pyplot as plt

st.set_page_config(page_title="Riesgo de abandono", page_icon="📉")
modelo = joblib.load("modelo_churn.joblib")

st.title("📉 Probabilidad de que un cliente abandone")
st.write("""**¿Qué hace?** Usa una *regresión logística* entrenada con el dataset Telco Customer Churn
para estimar la probabilidad de que un cliente cancele su servicio.
**¿Cómo usarla?** Ingresa los datos del cliente en la barra lateral y mira el resultado y el gráfico.""")

# ---- Entradas del usuario ----
s = st.sidebar
s.header("Datos del cliente")
tenure = s.slider("Antigüedad (meses)", 0, 72, 12)
monthly = s.slider("Cargo mensual (USD)", 18, 120, 70)
contract = s.selectbox("Contrato", ["Month-to-month", "One year", "Two year"])
internet = s.selectbox("Servicio de internet", ["DSL", "Fiber optic", "No"])
tech = s.selectbox("Soporte técnico", ["No", "Yes", "No internet service"])
pay = s.selectbox("Método de pago", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
paperless = s.selectbox("Factura electrónica", ["Yes", "No"])
senior = s.selectbox("Adulto mayor (65+)", ["No", "Si"])

cliente = pd.DataFrame([{"tenure": tenure, "MonthlyCharges": monthly, "Contract": contract,
                         "InternetService": internet, "TechSupport": tech, "PaymentMethod": pay,
                         "PaperlessBilling": paperless, "SeniorCitizen": senior}])

# ---- Predicción ----
p = modelo.predict_proba(cliente)[0, 1]
nivel = "ALTO" if p >= 0.6 else "MEDIO" if p >= 0.4 else "BAJO"
color = {"ALTO": "#d64545", "MEDIO": "#e8a33d", "BAJO": "#2a9d5c"}[nivel]

c1, c2 = st.columns(2)
c1.metric("Probabilidad de abandono", f"{p:.1%}")
c2.markdown(f"### Riesgo: <span style='color:{color}'>{nivel}</span>", unsafe_allow_html=True)
if nivel == "ALTO":
    st.warning("Cliente prioritario para una campaña de retención (descuento, contrato anual, soporte).")
elif nivel == "MEDIO":
    st.info("Vigilar al cliente y considerar una oferta de fidelización.")
else:
    st.success("Cliente con bajo riesgo de abandono.")

# ---- Gráfico 1: ¿qué variables empujan el riesgo? ----
x = modelo["prep"].transform(cliente)
x = x.toarray() if hasattr(x, "toarray") else x
contrib = pd.Series(x[0] * modelo["lr"].coef_[0], index=modelo["prep"].get_feature_names_out())
contrib = contrib[contrib != 0].sort_values()
contrib.index = [i.split("__")[1] for i in contrib.index]

fig, ax = plt.subplots(figsize=(7, 4))
ax.barh(contrib.index, contrib.values, color=["#d64545" if v > 0 else "#2a9d5c" for v in contrib])
ax.axvline(0, color="black", lw=0.8)
ax.set_title("¿Qué factores suben (rojo) o bajan (verde) el riesgo de este cliente?")
st.pyplot(fig)
st.caption("Cada barra es el aporte de esa característica al log-odds de abandono (coeficiente × valor del cliente).")

# ---- Gráfico 2: probabilidad según antigüedad ----
st.subheader("Riesgo según la antigüedad")
meses = list(range(0, 73, 3))
curva = pd.concat([cliente.assign(tenure=m) for m in meses])
fig2, ax2 = plt.subplots(figsize=(7, 3))
ax2.plot(meses, modelo.predict_proba(curva)[:, 1], color="#2a6fdb")
ax2.scatter([tenure], [p], color="red", zorder=3, label="Este cliente"); ax2.legend()
ax2.set_xlabel("Antigüedad (meses)"); ax2.set_ylabel("P(abandono)")
st.pyplot(fig2)
