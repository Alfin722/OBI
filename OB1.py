import streamlit as st
import numpy as np
import pandas as pd
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import os

genai.configure(api_key="AIzaSyBqCSiXefqi3oPq4IW1e_sSAm2skMw8Zkk")

st.set_page_config(page_title="OBI", layout="wide")
st.title("🧮 OBI: Optimalisasi Bisnis Indonesia")

def initialize_tableau(c, A, b, maximize=True, signs=None, M=1e6):
    n_constraints, n_vars = A.shape
    if signs is None:
        signs = ["<="] * n_constraints

    tableau = []
    slack_vars = []
    art_vars = []
    basis = []

    for i in range(n_constraints):
        row = list(A[i])

        slack = [0] * n_constraints
        art = [0] * n_constraints

        if signs[i] == "<=":
            slack[i] = 1
            slack_vars.append(f"S{i+1}")
            basis.append(f"S{i+1}")
        elif signs[i] == ">=":
            slack[i] = -1
            slack_vars.append(f"S{i+1}")
            art[i] = 1
            art_vars.append(f"A{i+1}")
            basis.append(f"A{i+1}")
        elif signs[i] == "=":
            art[i] = 1
            art_vars.append(f"A{i+1}")
            basis.append(f"A{i+1}")

        row += slack
        row += art
        row.append(b[i])
        tableau.append(row)

    # Objective function
    obj = list(-c if maximize else c)
    obj += [0] * n_constraints  # slack
    obj += [0] * n_constraints  # artificial
    obj += [0]  # RHS

    obj = np.array(obj, dtype=float)
    for i in range(n_constraints):
        if signs[i] in (">=", "="):
            penalty = (M if maximize else -M) * np.array(tableau[i], dtype=float)
            obj -= penalty

    tableau.append(list(obj))

    columns = [f"X{j+1}" for j in range(len(c))] + \
              [f"S{i+1}" for i in range(n_constraints)] + \
              [f"A{i+1}" for i in range(n_constraints)] + ["RHS"]

    return pd.DataFrame(tableau, columns=columns), basis


def pivot(df, row, col):
    pivot_val = df.iat[row, col]
    df.iloc[row] = df.iloc[row] / pivot_val
    for i in df.index:
        if i != row:
            df.iloc[i] = df.iloc[i] - df.iat[i, col] * df.iloc[row]
    return df


def solve_simplex(df, basis):
    history = []
    it = 0
    while True:
        it += 1
        obj = df.iloc[-1, :-1]
        if all(val >= -1e-8 for val in obj):
            break
        col = obj.values.argmin()
        ratios = []
        for i in range(len(df)-1):
            val = df.iat[i, col]
            rhs = df.iat[i, -1]
            ratios.append(rhs/val if val > 0 else np.inf)
        row = int(np.argmin(ratios))
        if ratios[row] == np.inf:
            st.error("Masalah tidak terbatas, solusi tidak diperoleh.")
            return None, None, history
        history.append((it, row, col, df.copy(), basis.copy()))
        df = pivot(df, row, col)
        basis[row] = df.columns[col]
    return df, basis, history


def get_gemini_suggestion(names, x, Z, constraints):
    prompt = (
        "\nNama anda adalah OBI, Anda berperan sebagai Business Analyst dan Strategi Bisnis UMKM di Indonesia.\n"
        "Berikut adalah output optimal dari model optimisasi laba UMKM Anda:\n"
        f"- Jenis Produk dan Jumlah (X): {list(zip(names, x.tolist()))}\n"
        f"- Nilai Laba Optimal (Z): {Z}\n\n"
        "Model optimisasi menggunakan:\n"
        "• Fungsi Tujuan: Maksimalkan Z = f(X) (laba total)\n"
        "• Fungsi Pembatas:\n"
    )
    for c in constraints:
        prompt += f"  - {c}\n"
    prompt += (
        "Berdasarkan data di atas, tolong berikan:\n"
        "1. **Interpretasi Variabel**\n"
        "   – Jelaskan makna tiap variabel X dalam konteks operasional UMKM.\n"
        "2. **Analisis Hasil**\n"
        "   – Uraikan arti nilai Z dan bandingkan dengan target atau kondisi saat ini.\n"
        "3. **Analisis Sensitivitas**\n"
        "   – Simulasikan dampak perubahan 10–20% pada satu sumber daya.\n"
        "4. **Strategi Peningkatan**\n"
        "   – Rekomendasikan langkah konkret untuk meningkatkan Z.\n"
        "5. **Mitigasi Risiko**\n"
        "   – Identifikasi risiko dan cara mitigasinya.\n"
        "6. **Kesimpulan**\n"
        "   – Ringkas poin-poin utama dan langkah selanjutnya.\n\n"
        "Susun jawaban secara terstruktur, ringkas, dan aplikatif bagi pelaku UMKM di Indonesia."
    )
    model = genai.GenerativeModel("gemini-2.0-flash")
    chat = model.start_chat(history=[])
    response = chat.send_message(prompt)
    return response.text.strip()

with st.sidebar:
    st.header("Model LP")
    mode = st.selectbox("Tipe Operasi", ["Maksimasi", "Minimasi"])
    maximize = (mode == "Maksimasi")
    n_vars = st.number_input("Jumlah Produk", min_value=1, value=3, step=1)
    n_cons = st.number_input("Jumlah Kendala", min_value=1, value=2, step=1)

tab1, tab2 = st.tabs([
    "🧮 Optimalisasi menggunakan Linear Programming",
    "🤖 Konsultasi UMKM dengan OBI"
])

with tab1:
    st.header("🧮 Optimalisasi menggunakan Linear Programming")
    st.markdown("Gunakan aplikasi ini untuk optimalisasi bisnis anda, agar memperoleh laba maksimal.")
    
    col1, col2 = st.columns(2)
    with col1:
        # Input nama produk
        st.subheader("Jenis Produk")
        names = []
        for j in range(n_vars):
            names.append(st.text_input(f"Nama produk {j+1}", value=f"Produk {j+1}", key=f"name{j}"))
    with col2:
        # Input keuntungan per produk
        st.subheader("Keuntungan Per produk")
        c = []
        for j in range(n_vars):
            c.append(st.number_input(f"Keuntungan {names[j]}", value=0.0, key=f"c{j}"))
        c = np.array(c)
    # Kendala
    st.subheader("Kendala")
    A = np.zeros((n_cons, n_vars))
    b = np.zeros(n_cons)
    signs = []
    for i in range(n_cons):
        with st.expander(f"Kendala {i+1}", expanded=False):
            row_cols = st.columns([*([1]*n_vars), .5, .5])
            for j in range(n_vars):
                A[i,j] = row_cols[j].number_input(f"bahan {names[j]}", key=(i,j))
            signs.append(row_cols[-2].selectbox("Tanda", ["<=", ">=", "="], key=f"s{i}"))
            b[i] = row_cols[-1].number_input("Biaya", key=f"b{i}")

    if st.button("Hitung Optimal"):
        df, basis = initialize_tableau(c, A, b, maximize, signs)
        st.subheader("Tabel Simpleks Awal")
        st.dataframe(df)

        result, basis, history = solve_simplex(df.copy(), basis)
        if result is not None:
            st.success("Solusi optimal ditemukan!")
            st.subheader("Tabel Simpleks Akhir")
            st.dataframe(result)

            with st.expander("Langkah Iterasi"):
                for it, row, col, tab, bas in history:
                    st.markdown(f"**Iterasi {it}**: pivot di baris {row+1}, kolom {col+1}, basis: {bas}")
                    st.dataframe(tab)

            n_vars = len(c)
            x = np.zeros(n_vars)
            for i, var in enumerate(basis):
                if var.startswith("X"):
                    idx = int(var[1:]) - 1
                    x[idx] = result.iat[i, -1]
            sol = pd.DataFrame({"Produk": names + ["Keuntungan maksimal"],
                                "Nilai": list(x) + [result.iat[-1,-1]]})
            st.subheader("Hasil Solusi")
            st.table(sol)

            csv = sol.to_csv(index=False).encode('utf-8')
            st.download_button("Download Hasil CSV", csv, "solusi.csv", "text/csv")
            constraints = []
            for i in range(n_cons):
                terms = [f"{A[i,j]}{names[j]}" for j in range(n_vars) if A[i,j] != 0] + []
                lhs = " + ".join(terms) if terms else "0"
                constraints.append(f"{lhs} {signs[i]} {b[i]}")

            with st.spinner("Mengambil saran dari OBI…"):
                suggestion = get_gemini_suggestion(names, x, result.iat[-1, -1], constraints)

            st.markdown("---")
            st.subheader("💡 Saran OBI")
            st.markdown(suggestion)

with tab2:
    st.header("🤖 Konsultasi UMKM dengan OBI")
    st.markdown("Lakukan konsultasi UMKM dengan OBI.")

    bubble_css = """
    <style>
      .chat-container {
          display: flex;
          flex-direction: column;
          margin-top: 20px;
      }
      .chat-bubble {
          max-width: 70%;
          padding: 10px 15px;
          border-radius: 20px;
          margin: 5px;
          font-size: 16px;
          line-height: 1.4;
      }
      .user-bubble {
          background-color: #1E90FF;
          align-self: flex-end;
          border-bottom-right-radius: 0;
      }
      .ai-bubble {
          background-color: #081F5C;
          align-self: flex-start;
          border-bottom-left-radius: 0;
      }
    </style>
    """
    st.markdown(bubble_css, unsafe_allow_html=True)

    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    for chat in st.session_state["chat_history"]:
        role = chat["role"]
        message = chat["message"]
        if role == "user":
            bubble = f'<div class="chat-bubble user-bubble"><strong>Anda:</strong> {message}</div>'
        else:
            bubble = f'<div class="chat-bubble ai-bubble"><strong>OBI:</strong> {message}</div>'
        st.markdown(bubble, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    user_input = st.text_input("Tanyakan sesuatu tentang UMKM atau bisnis", key="user_input")

    if st.button("Kirim") and user_input:
        st.session_state["chat_history"].append({"role": "user", "message": user_input})

        def get_ai_response(prompt):
            try:
                model = genai.GenerativeModel(
                    model_name='models/gemini-2.0-flash',
                    system_instruction="Nama kamu adalah OBI, kamu berperan sebagai ahli business analyst dan Strategi Bisnis UMKM di Indonesia."
                )
                response = model.generate_content(
                    prompt,
                    generation_config=genai.types.GenerationConfig(
                        temperature=0.7,
                        max_output_tokens=2048
                    ),
                    safety_settings={
                        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE
                    }
                )
                return response.text
            except Exception as e:
                return f"Terjadi kesalahan: {str(e)}"

        ai_reply = get_ai_response(user_input)
        st.session_state["chat_history"].append({"role": "assistant", "message": ai_reply})
        st.rerun()

st.markdown("---")
st.markdown("© 2025 - OBI : Operasi Bisnis Indonesia | Math Wizard I")