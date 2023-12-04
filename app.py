import chainlit
import torch
from langchain.chains import LLMChain, RetrievalQA
from langchain.llms.huggingface_pipeline import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from transformers import GenerationConfig, pipeline, AutoModelForCausalLM, AutoTokenizer, AutoConfig

# from langchain.llms import

prompt_template = '''[INST] <<SYS>>
You are a helpful, respectful and honest assistant in Leap-Gaming company.
Always generate grammatically correct sentences.
Always answer as helpfully as possible, while being safe.
Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content.
Please ensure that your responses are socially unbiased and positive in nature.
If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct.
If you don't know the answer to a question, please don't share false information.
This is a conversation between you as support team worker and client which name is {name}
Your superior email is superior@leap-gaming.com pleas ask client contact him if you don't have enough information for help client
<</SYS>>
{context}
{conversation}
[/INST]
'''


def load_model():
    config = AutoConfig.from_pretrained("TheBloke/Llama-2-7B-Chat-GPTQ")
    config.quantization_config["use_exllama"] = False
    # config.quantization_config["disable_exllama"] = True
    config.quantization_config["exllama_config"] = {"version": 2}

    model = AutoModelForCausalLM.from_pretrained(
        "TheBloke/Llama-2-7B-Chat-GPTQ",
        trust_remote_code=True,
        torch_dtype=torch.float32,
        device_map="cpu",
        cache_dir="models_cache",
        revision="gptq-4bit-64g-actorder_True",
        config=config
    )
    model.to_bettertransformer()

    tokenizer = AutoTokenizer.from_pretrained("TheBloke/Llama-2-7B-Chat-GPTQ", cache_dir="models_cache")

    generation_config = GenerationConfig.from_pretrained("TheBloke/Llama-2-7B-Chat-GPTQ", cache_dir="models_cache")
    generation_config.max_new_tokens = 1024
    generation_config.temperature = 0.02
    generation_config.top_p = 0.85
    generation_config.top_k = 33
    generation_config.do_sample = True
    generation_config.repetition_penalty = 1.15

    text_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        return_full_text=True,
        generation_config=generation_config
    )

    llm = HuggingFacePipeline(pipeline=text_pipeline, model_kwargs={"temperature": 0})
    return llm


def prepare_prompt():
    return PromptTemplate(
        input_variables=["name", "conversation", "context"],
        template=prompt_template,
    )


def create_llm_chain(llm, prompt):
    # return RetrievalQA.from_chain_type(
    #     llm=llm,
    #     chain_type="stuff",
    #     return_source_documents=False,
    #     chain_type_kwargs={"prompt": prompt},
    #     retriever=
    # )
    return LLMChain(llm=llm, prompt=prompt)


@chainlit.on_chat_start
async def main():
    user = ""
    client_id = ""
    while user == "":
        response = await chainlit.AskUserMessage(author="Assistant",
                                                 content="Hi, Greetings from the Leap-Gaming support team. What is your name?",
                                                 timeout=10).send()
        if response:
            user = response['content']
            break
        else:
            await chainlit.Message(author="Assistant", content="Sorry, I didn't get that. Please try again.").send()
    chainlit.user_session.set("user_name", user)
    while client_id == "":
        response = await chainlit.AskUserMessage(author="Assistant",
                                                 content="What is your client ID?", timeout=10).send()
        if response:
            await chainlit.Message(author="Assistant",
                                   content="Processing...").send()
            client_id = response['content']
            break
        else:
            await chainlit.Message(author="Assistant", content="Sorry, I didn't get that. Please try again.").send()
    chainlit.user_session.set("client_id", client_id)
    chainlit.user_session.set("llm_chain", create_llm_chain(load_model(), prepare_prompt()))
    await chainlit.Message(author="Assistant",
                           content="Hi, how can we help you today?").send()
    chainlit.user_session.set("context", "I: Hi, how can we help you today")


@chainlit.on_message
async def msg(message: chainlit.Message):
    user = chainlit.user_session.get("user_name")
    context = chainlit.user_session.get("context")
    chain = chainlit.user_session.get("llm_chain")
    callback = chainlit.AsyncLangchainCallbackHandler(
        answer_prefix_tokens=["FINAL", "ANSWER"]
    )
    context += f"\n{user}:" + message.content
    callback.answer_reached = True
    answer = await chain.acall({'name': user, 'context': context, 'conversation': message.content}, callbacks=[callback])
    print(answer)
    context += "\nI: " + answer.get("text")
    chainlit.user_session.set("context", context)
    await chainlit.Message(content=answer.get("text")).send()

