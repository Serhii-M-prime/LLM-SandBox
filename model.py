# from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
#
#
# class PreparedModel:
#     model_name_or_path = "TheBloke/Llama-2-7B-Chat-GPTQ"
#     cache_dir = 'models_cache'
#     revision = "gptq-4bit-64g-actorder_True"
#
#     @staticmethod
#     def get_model():
#         model = AutoModelForCausalLM.from_pretrained(PreparedModel.model_name_or_path,
#                                                      device_map="auto",
#                                                      model_type="llama",
#                                                      trust_remote_code=False,
#                                                      cache_dir=PreparedModel.cache_dir,
#                                                      revision=PreparedModel.revision)
#         tokenizer = AutoTokenizer.from_pretrained(PreparedModel.model_name_or_path)
#         pipe = pipeline(
#             "text-generation",
#             model=model,
#             tokenizer=tokenizer,
#             max_new_tokens=1024,
#             do_sample=True,
#             temperature=0.3,
#             top_p=0.95,
#             top_k=40,
#             repetition_penalty=1.1
#         )
#         return pipe