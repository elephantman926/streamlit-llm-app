import os

import streamlit as st
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage


# =========================
# 環境変数の読み込み
# =========================
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


# =========================
# LLMに問い合わせる関数
# =========================
def ask_llm(user_input: str, expert_type: str) -> str:
    """
    入力テキストと専門家タイプを受け取り、
    LLMからの回答を返す関数
    """

    # 専門家ごとのシステムメッセージ
    if expert_type == "ITエンジニア":
        system_message = (
            "あなたは経験豊富なITエンジニアです。"
            "技術的に正確で、初心者にも分かりやすく説明してください。"
        )
    elif expert_type == "ビジネスコンサルタント":
        system_message = (
            "あなたは優秀なビジネスコンサルタントです。"
            "ビジネス視点で、要点を整理して説明してください。"
        )
    else:
        system_message = "あなたは親切な専門家です。"

    # LLMの初期化
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.7,
        api_key=OPENAI_API_KEY,
    )

    messages = [
        SystemMessage(content=system_message),
        HumanMessage(content=user_input),
    ]

    response = llm(messages)
    return response.content


# =========================
# Streamlit UI
# =========================
st.set_page_config(page_title="LLM専門家相談アプリ", layout="centered")

st.title("🤖 LLM専門家相談アプリ")

st.markdown(
    """
このWebアプリは、**Streamlit** と **LangChain** を利用した  
LLM（大規模言語モデル）搭載のサンプルアプリです。

### 使い方
1. 専門家の種類を選択します  
2. 質問したい内容を入力します  
3. 「送信」ボタンを押すと、LLMが回答します
"""
)

# 専門家選択（ラジオボタン）
expert_type = st.radio(
    "専門家の種類を選択してください",
    ["ITエンジニア", "ビジネスコンサルタント"],
)

# 入力フォーム
user_input = st.text_area(
    "質問を入力してください",
    placeholder="例：Streamlitとは何ですか？",
)

# 送信ボタン
if st.button("送信"):
    if not user_input.strip():
        st.warning("質問を入力してください。")
    else:
        with st.spinner("LLMが回答中です..."):
            answer = ask_llm(user_input, expert_type)

        st.subheader("💡 回答結果")
        st.write(answer)
