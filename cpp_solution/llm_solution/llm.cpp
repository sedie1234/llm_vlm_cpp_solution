#include <onnxruntime_cxx_api.h>
#include <tokenizers_cpp.h>

#include <iostream>
#include <fstream>
#include <string>
#include <vector>

// definitions
#define TEST_PROMPT "<bos><start_of_turn>user\nYou are a helpful assistant."\
                    "\n\nWrite me a short poem about Machine Learning."\
                    "<end_of_turn>\n<start_of_turn>model"
#define VOCAB_SIZE      262144
#define MAX_NEW_TOKEN   128
#define EOS_TOKEN_ID    106
#define DECODER_NUM     26
#define BATCH_SIZE      1
#define KEY_VALUE_NUM   1


std::string LoadBytesFromFile(const std::string& path);
int64_t Argmax(const float* data, int64_t size);

int main(int argc, char** argv){


    // 1. load stage : model, config, processor, tokenizer
    // argument parsing
    if(argc != 3){
        printf("Usage : ./run_llm <model_path> <tokenizer_path> \n");
        return -1;
    }

    std::string model_path(argv[1]);
    std::string tokenizer_path(argv[2]);

    // load model, open session
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "llm_infer");
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);

    Ort::Session session(env, model_path.c_str(), session_options);
    Ort::AllocatorWithDefaultOptions allocator;

    // create tokenizer obj
    //auto tok = Tokenizer::FromFile(tokenizer_path);
    auto blob = LoadBytesFromFile(tokenizer_path);
    auto tok = tokenizers::Tokenizer::FromBlobJSON(blob);


    // 2. prepare input stage
    // prompt encode
    std::string prompt = TEST_PROMPT;
    std::vector<int> input_tok = tok->Encode(prompt);
    std::vector<int64_t> input_ids;
    input_ids.reserve(input_tok.size());

    for(int id : input_tok){
        input_ids.push_back(static_cast<int64_t>(id));
    }

    // create memory info
    Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    // get and set input name, output name
    std::vector<std::string> input_names_str;
    std::vector<const char*> input_names;
    std::vector<std::string> output_names_str;
    std::vector<const char*> output_names;
    
    // set name of inputs
    size_t num_inputs = session.GetInputCount();
    input_names_str.reserve(num_inputs);
    input_names.reserve(num_inputs);
    for (size_t i = 0; i < num_inputs; ++i) {
        auto name = session.GetInputNameAllocated(i, allocator);
        input_names_str.push_back(name.get());
        input_names.push_back(input_names_str.back().c_str());
        // std::cout << "input[" << i << "] = " << name.get() << std::endl;
    }
    
    //set name of outputs
    size_t num_outputs = session.GetOutputCount();
    output_names_str.reserve(num_outputs);
    output_names.reserve(num_outputs);
    for (size_t i = 0; i < num_outputs; ++i) {
        auto name = session.GetOutputNameAllocated(i, allocator);
        output_names_str.push_back(name.get());
        output_names.push_back(output_names_str.back().c_str());
        // std::cout << "output[" << i << "] = " << name.get() << std::endl;
    }
    

    // 3. generation stage
    std::vector<int64_t> generated_tokens;

    // set init past_key_values
    std::vector<Ort::Value> past_key_values;   
    std::vector<int64_t> past_key_value_shape = {BATCH_SIZE, KEY_VALUE_NUM, 0, 256};

    // create init past_key_values tensor
    for(int i=0; i<DECODER_NUM*2; i++){
        Ort::Value empty_key_value = Ort::Value::CreateTensor<float>(
            mem_info, nullptr, 0,
            past_key_value_shape.data(), past_key_value_shape.size()
        );
        past_key_values.push_back(std::move(empty_key_value));
    }

    // set init input shape
    std::vector<int64_t> input_shape = {BATCH_SIZE, static_cast<int64_t>(input_ids.size())};
    
    // create init input tensor
    Ort::Value input_tensor = Ort::Value::CreateTensor<int64_t>(
        mem_info, input_ids.data(), input_ids.size(),
        input_shape.data(), input_shape.size()
    );

    // set init position_ids
    std::vector<int64_t> position_ids;
    // position_ids.reserve(generated_ids.size());
    for(int64_t i=0; i<input_ids.size(); i++){
        position_ids.push_back(i+1);
    }

    // create init position_ids tensor
    Ort::Value position_ids_tensor = Ort::Value::CreateTensor<int64_t>(
        mem_info, position_ids.data(), position_ids.size(),
        input_shape.data(), input_shape.size()
    );

    // shape for generation loop
    input_shape = {BATCH_SIZE, 1};

    // generation loop
    for(int i=0; i<MAX_NEW_TOKEN; i++){
        // std::cout << "Step : " << i ;
        // std::cout << " \t ";
        // << std::endl;

        // set input
        std::vector<Ort::Value> input_tensors;
        input_tensors.push_back(std::move(input_tensor));
        input_tensors.push_back(std::move(position_ids_tensor));
        
        for(int i=0; i<DECODER_NUM*2; i++){
            input_tensors.push_back(std::move(past_key_values[i]));
        }

        // run inference
        auto outputs = session.Run(Ort::RunOptions{nullptr},
            input_names.data(), input_tensors.data(), input_tensors.size(), 
            output_names.data(), output_names.size());

        // get output tensor
        // float* logits = outputs[0].GetTensorMutableData<float>();
        // int64_t offset = (input_ids.size() + generated_tokens.size() - 1) * VOCAB_SIZE;
        // int64_t next_token_id = Argmax(logits + offset, VOCAB_SIZE);
        
        auto shape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();  // [1, seq_len, vocab]
        int64_t vocab_size = shape[2];
        int64_t last_token_index = shape[1] - 1;
        
        float* logits = outputs[0].GetTensorMutableData<float>();
        int64_t next_token_id = Argmax(logits + last_token_index * VOCAB_SIZE, VOCAB_SIZE);

        int64_t position_id = generated_tokens.size()+1;

        // Append and continue
        generated_tokens.push_back(next_token_id);
        
        // Stop if <eos>
        if (next_token_id == EOS_TOKEN_ID) break;
         
        std::vector<Ort::Value> present_key_values;
        for (int i = 0; i < DECODER_NUM * 2; ++i) {
            present_key_values.push_back(std::move(outputs[i + 1]));
        }
        past_key_values = std::move(present_key_values); 

        // create input tensor
        input_tensor = Ort::Value::CreateTensor<int64_t>(
            mem_info, &next_token_id, 1,
            input_shape.data(), input_shape.size()
        );

        // create position_ids tensor
        position_ids_tensor = Ort::Value::CreateTensor<int64_t>(
            mem_info, &position_id, 1,
            input_shape.data(), input_shape.size()
        );
        
    }

    std::vector<int> generated_ids;
    generated_ids.reserve(generated_tokens.size());
    // convert int64_t to int
    for(int64_t token : generated_tokens){
        generated_ids.push_back(static_cast<int>(token));
    }

    std::cout << "Generated tokens : ";
    for(int i=0; i<10; i++){
        std::cout << generated_tokens[i] << " ";
    }
    std::cout << std::endl;

    // 4. decode output tokens, print result
    std::string decoded_output = tok->Decode(generated_ids);
    std::cout << "Answer : " << decoded_output << std::endl;


    return 0;
}


std::string LoadBytesFromFile(const std::string& path) {
    std::ifstream fs(path, std::ios::in | std::ios::binary);
    if (fs.fail()) {
      std::cerr << "Cannot open " << path << std::endl;
      exit(1);
    }
    std::string data;
    fs.seekg(0, std::ios::end);
    size_t size = static_cast<size_t>(fs.tellg());
    fs.seekg(0, std::ios::beg);
    data.resize(size);
    fs.read(data.data(), size);
    return data;
  }

int64_t Argmax(const float* data, int64_t size) {
    int64_t max_index = 0;
    float max_value = data[0];
    for (int64_t i = 1; i < size; ++i) {
        if (data[i] > max_value) {
            max_value = data[i];
            max_index = i;
        }
    }
    return max_index;
}
