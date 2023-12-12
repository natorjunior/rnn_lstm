
# Página inicial do aplicativo
def page_one(st):
    st.title("RNN")
    st.image('fig1.png')
    #st.session_state.page_num = 2
    col1,col2,col3,col4 = st.columns(4)
    with col4:
        if st.button("Proximo"):
            # Quando o botão "Next" é clicado, mude para a próxima página
            st.session_state.page_num = 2
            st.session_state.page_num = 2
            

# Página seguinte do aplicativo
def page_two(st):
        st.title("LSTM  - 1")
        st.image('lstm_fig1.png')
        #st.session_state.page_num = 1
        col1,col2,col3,col4 = st.columns(4)
        with col1:
            if st.button("anterior"):
                st.session_state.page_num = 1
        with col4:
            if st.button("Proximo"):
                st.session_state.page_num = 3
def page_tres(st):
        st.title("LSTM - 2 ")
        col1,col2,col3,col4 = st.columns(4)
        st.image('lstm_fig2.png')
        with col1:
            if st.button("anterior"):
                st.session_state.page_num = 2
        with col4:
            if st.button("Proximo"):
                st.session_state.page_num = 4
def page_4(st):
        st.title("LSTM - 3")
        col1,col2,col3,col4 = st.columns(4)
        st.image('lstm_fig3.png')
        with col1:
            if st.button("anterior"):
                st.session_state.page_num = 3
        with col4:
            if st.button("Proximo"):
                st.session_state.page_num = 5
def page_5(st):
        st.title("LSTM - 4")
        col1,col2,col3,col4 = st.columns(4)
        st.image('lstm_fig4.png')
        with col1:
            if st.button("anterior"):
                st.session_state.page_num = 4
        with col4:
            if st.button("Proximo"):
                st.session_state.page_num = 6
def page_6(st):
        st.title("LSTM - 5")
        col1,col2,col3,col4 = st.columns(4)
        st.image('lstm_fig5.png')
        with col1:
            if st.button("anterior"):
                st.session_state.page_num = 5
        with col4:
            if st.button("Proximo"):
                st.session_state.page_num = 7

def page_7(st):
        st.title("LSTM - 6")
        col1,col2,col3,col4 = st.columns(4)
        st.image('lstm_fig6.png')
        with col1:
            if st.button("anterior"):
                st.session_state.page_num = 6
