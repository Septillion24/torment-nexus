import os
import lief
import gradio as gr

def process_file(binary_location, number):
    binary = lief.PE.parse(binary_location)
    data_directory = binary.data_directories 
    new_address = int(number)
    data_directory[4].rva = new_address
    binary.write("/workspaces/torment-nexus/binaries/mimikatz-edited-gradio.exe")

def main():
    with gr.Blocks() as demo:
        file_input = gr.File(label="Upload your file")
        number_input = gr.Number(label="Enter a number")
        submit_button = gr.Button("Submit")
        output = gr.Textbox(label="Output")

        submit_button.click(fn=process_file, inputs=[file_input, number_input], outputs=output)

    demo.launch()

if __name__ == "__main__":
    main()
