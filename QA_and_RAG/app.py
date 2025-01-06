from functools import lru_cache
import gradio as gr

from QA_and_RAG.src.utils.chat_utils import ModelManager
from QA_and_RAG.src.utils.utilities import UISettings, upload_file
from QA_and_RAG.src.chatbot import ChatType


@lru_cache(maxsize=1)
def create_qa_interface() -> gr.Blocks:
    """Creates a Q&A interface with chat and file processing capabilities.

    Returns
    -------
    gr.Blocks
        The Gradio Blocks interface containing all UI components and event handlers.
    """
    bot_instance: ModelManager = ModelManager()

    with gr.Blocks() as demo:
        with gr.Tabs():
            with gr.TabItem("Q&A-and-RAG-with-SQL-and-TabularData"):
                ##############
                # First ROW:
                ##############
                with gr.Row() as row_one:
                    chatbot: gr.Chatbot = gr.Chatbot(
                        [],
                        type="messages",
                        elem_id="chatbot",
                        bubble_full_width=False,
                        height=500,
                        avatar_images=(("images/AI-agent.jpg"), "images/openai.png"),
                    )
                    # **Adding like/dislike icons
                    chatbot.like(UISettings.feedback, None, None)
                ##############
                # SECOND ROW:
                ##############
                with gr.Row():
                    input_txt: gr.Textbox = gr.Textbox(
                        lines=4,
                        scale=8,
                        placeholder="Enter text and press enter",
                        container=False,
                    )
                ##############
                # Third ROW:
                ##############
                with gr.Row() as row_two:
                    text_submit_btn: gr.Button = gr.Button(value="Submit text")
                    upload_btn: gr.UploadButton = gr.UploadButton(
                        "📁 Upload CSV files",
                        file_types=[".csv"],
                        file_count="multiple",
                    )
                    app_functionality: gr.Dropdown = gr.Dropdown(
                        label="App functionality",
                        choices=["Chat", "Process files"],
                        value="Chat",
                    )
                    chat_type: gr.Dropdown = gr.Dropdown(
                        label="Chat type",
                        choices=[
                            ChatType.QA_WITH_STORED_SQL_DB.value,
                            ChatType.QA_WITH_STORED_FLAT_FILE_SQL_DB.value,
                            ChatType.QA_WITH_UPLOADED_FLAT_FILE_SQL_DB.value,
                        ],
                        value=ChatType.QA_WITH_STORED_SQL_DB,
                    )
                    clear_button: gr.ClearButton = gr.ClearButton([input_txt, chatbot])
                ##############
                # Process:
                ##############
                file_msg: gr.EventData = upload_btn.upload(
                    fn=upload_file,
                    inputs=[upload_btn, chatbot, app_functionality],
                    outputs=[input_txt, chatbot],
                    queue=False,
                )

                txt_msg: gr.EventData = input_txt.submit(
                    fn=bot_instance.get_response,
                    inputs=[chatbot, input_txt, chat_type, app_functionality],
                    outputs=[input_txt, chatbot],
                    queue=False,
                ).then(
                    lambda: gr.Textbox(interactive=True), None, [input_txt], queue=False
                )

                txt_msg: gr.EventData = text_submit_btn.click(
                    fn=bot_instance.get_response,
                    inputs=[chatbot, input_txt, chat_type, app_functionality],
                    outputs=[input_txt, chatbot],
                    queue=False,
                ).then(
                    lambda: gr.Textbox(interactive=True), None, [input_txt], queue=False
                )

    return demo


if __name__ == "__main__":
    demo = create_qa_interface()
    demo.launch(debug=True)
