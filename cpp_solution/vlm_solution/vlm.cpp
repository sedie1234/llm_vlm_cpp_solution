#include <onnxruntime_cxx_api.h>
#include <tokenizers_cpp.h>
#include <opencv2/opencv.hpp>

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <random>
#include <algorithm>
#include <numeric>
#include <cmath>

// definitions
// #define TEST_PROMPT "<|im_start|>system\nYou are a helpful assistant."\
//                     "<|im_end|>\n<|im_start|>user\nWhere do you think this image is from?"\
//                     "<|im_end|>\n<|im_start|>assistant"
#define TEST_PROMPT "<|im_start|>user\n<image>\nWhere was this photo taken?<|im_end|>"\
                    "\n<|im_start|>assistant\n"
                    
#define VOCAB_SIZE          151646
#define MAX_NEW_TOKEN       128
#define EOS_TOKEN_ID        106
#define DECODER_NUM         24
#define BATCH_SIZE          1
#define KEY_VALUE_NUM       2
#define IMAGE_TOKEN_INDEX   151646
#define USE_SAMPLING        false
#define TOP_P               0.99f


//structures
typedef struct {
    int crop_size[2];
    bool do_center_crop;
    bool do_convert_rgb;
    bool do_normalize;
    bool do_rescale;
    bool do_resize;
    double  image_mean[3];
    double  image_std[3];
    double rescale_factor;
    std::pair<std::string, int> size;
    int resample;
}ImgProcessorConfig;

typedef struct {
    std::string input_text;
    std::string image_path;
    std::string tokenizer_path;
}Arguments;

typedef struct {
    int64_t input_token_len;
    Ort::Value input_ids;
    std::vector<Ort::Value> past_key_values;
    std::vector<int64_t> generated_tokens;
}NextInput;

typedef struct {
    std::vector<std::string> input_names_str;
    std::vector<const char*> input_names;
    std::vector<std::string> output_names_str;
    std::vector<const char*> output_names;
}InputOutputNames;

// function declarations
float Half2Float(uint16_t value);
InputOutputNames GetInputOutputNames(Ort::Session& session, Ort::AllocatorWithDefaultOptions& allocator);
std::string LoadBytesFromFile(const std::string& path);
int64_t Argmax(const float* data, int64_t size);
std::pair<std::vector<float>, std::vector<int64_t>> ProcessImage(std::string image_path); //return processed image and its shape
int top_p_sampling(const std::vector<float>& logits, float top_p);
void prefill(Arguments& args, Ort::Session& image_emb_session, Ort::Session& text_emb_session, 
        Ort::Session& decoding_session, Ort::AllocatorWithDefaultOptions& allocator, Ort::MemoryInfo& mem_info,
        NextInput* next_input, tokenizers::Tokenizer& tok);
std::string decode(Arguments& args, Ort::Session& text_emb_session,
    Ort::Session& decoding_session, Ort::AllocatorWithDefaultOptions& allocator, 
    Ort::MemoryInfo& mem_info, NextInput* first_input, tokenizers::Tokenizer& tok);


// main
int main(int argc, char** argv) {
    if (argc != 6) {
        std::cerr << "Usage: ./run_vlm <image_encoder.onnx> <text_encoder.onnx>"
                  << "<decoder.onnx> <tokenizer.json> <image_path>" << std::endl;
        return 1;
    }

    std::string image_encoder_path = argv[1];
    std::string text_encoder_path = argv[2];
    std::string decoder_path = argv[3];
    std::string tokenizer_path = argv[4];
    std::string image_path = argv[5];

    // prompt
    std::string prompt = TEST_PROMPT;

    // Arguments
    Arguments args;
    args.input_text = prompt;
    args.image_path = image_path;
    args.tokenizer_path = tokenizer_path;

    // Tokenizer
    auto blob = LoadBytesFromFile(tokenizer_path);
    std::unique_ptr<tokenizers::Tokenizer> tok = tokenizers::Tokenizer::FromBlobJSON(blob);

    // ONNX Runtime 
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "vlm");
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    // Memory & Allocator
    Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::AllocatorWithDefaultOptions allocator;

    // 모델 로드
    Ort::Session image_emb_session(env, image_encoder_path.c_str(), session_options);
    Ort::Session text_emb_session(env, text_encoder_path.c_str(), session_options);
    Ort::Session decoding_session(env, decoder_path.c_str(), session_options);

    // Prefill 단계 실행
    NextInput first_input;
    prefill(args, image_emb_session, text_emb_session, decoding_session, allocator, mem_info, &first_input, *tok);
    std::cout << "prefill done" << std::endl;

    // Decode 단계 실행
    std::string result = decode(args, text_emb_session, decoding_session, allocator, mem_info, &first_input, *tok);

    // 출력
    std::cout << "Final generated text: " << result << std::endl;

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

std::pair<std::vector<float>, std::vector<int64_t>> ProcessImage(std::string image_path){

    // image processor config
    ImgProcessorConfig config;
    config.crop_size[0] = 224;
    config.crop_size[1] = 224;
    config.do_center_crop = true;
    config.do_convert_rgb = true;
    config.do_normalize = true;
    config.do_rescale = true;
    config.do_resize = true;
    config.image_mean[0] = 0.48145466;
    config.image_mean[1] = 0.4578275;
    config.image_mean[2] = 0.40821073;
    config.image_std[0] = 0.26862954;
    config.image_std[1] = 0.26130258;
    config.image_std[2] = 0.27577711;
    config.rescale_factor = 1.0 / 255.0;
    config.size = std::make_pair("shortest_edge", 224);
    config.resample = cv::INTER_CUBIC; // simlar to BICUBIC

    // load image
    cv::Mat image = cv::imread(image_path, cv::IMREAD_COLOR); //default : BGR
    if (image.empty()) {
        std::cerr << "Could not open or find the image!" << std::endl;
        return std::make_pair(std::vector<float>(), std::vector<int64_t>());
    }
    
    // convert to rgb
    if (config.do_convert_rgb)  cv::cvtColor(image, image, cv::COLOR_BGR2RGB);// convert BGR to RGB    

    // resize image
    if (config.do_resize) {
        int shortest_edge = image.cols < image.rows ? image.cols : image.rows;
        double scale_factor = config.size.second / static_cast<double>(shortest_edge);
        cv::resize(image, image, cv::Size(), scale_factor, scale_factor, config.resample);
    }
    
    // center crop
    if (config.do_center_crop) {
        float crop_x = (image.cols - config.crop_size[0]) / 2.0f;
        float crop_y = (image.rows - config.crop_size[1]) / 2.0f;
        cv::Rect roi((int)(crop_x), (int)(crop_y), config.crop_size[0], config.crop_size[1]);
        image = image(roi);
    }

    cv::imwrite("crop_rescaled_input_image.jpg", image);

// cv::dnn::blobFromImage(image, config.rescale_factor);???
    // convert to float
    image.convertTo(image, CV_32F);

    // rescale
    if (config.do_rescale) {
        image = image * config.rescale_factor;
    }

    // normalize
    if (config.do_normalize) {
        cv::Mat mean(config.crop_size[0], config.crop_size[1], CV_32FC3, cv::Scalar(config.image_mean[0], config.image_mean[1], config.image_mean[2]));
        cv::Mat std(config.crop_size[0], config.crop_size[1], CV_32FC3, cv::Scalar(config.image_std[0], config.image_std[1], config.image_std[2]));
        image = (image - mean);
        image = image / std;
    }
    
    // transpose image to (C, H, W)    
    std::vector<cv::Mat> channels(3);
    cv::split(image, channels);

    std::vector<float> image_data;
    for(int i=0; i<3; i++){
        const float* ptr = channels[i].ptr<float>(0);
        image_data.insert(image_data.end(), ptr, ptr + channels[i].total());
    }

    std::vector<int64_t> image_shape = {1, 3, config.crop_size[0], config.crop_size[1]};

    return std::make_pair(image_data, image_shape);
}

int top_p_sampling(const std::vector<float>& logits, float top_p){

    int vocab_size = logits.size();

    // sort indices based on logits
    std::vector<int> sorted_indices(vocab_size);
    std::iota(sorted_indices.begin(), sorted_indices.end(), 0);
    std::sort(sorted_indices.begin(), sorted_indices.end(),
              [&logits](int a, int b) { return logits[a] > logits[b]; });

    float max_logit = logits[sorted_indices[0]];

    // calculate cumulative probabilities
    std::vector<float> exp_logits(vocab_size);
    for(int i=0; i<vocab_size; i++){
        exp_logits[i] = std::exp(logits[sorted_indices[i]] - max_logit);
    }
    float sum_exp = std::accumulate(exp_logits.begin(), exp_logits.end(), 0.0f);

    std::vector<float> cumulative_probs(vocab_size);
    float cumulative_sum = 0.0f;
    for(int i=0; i<vocab_size; i++){
        cumulative_sum += exp_logits[i];
        cumulative_probs[i] = cumulative_sum / sum_exp;
    }
    
    // find cutoff index
    int cutoff_index = 0;
    for(; cutoff_index<vocab_size; cutoff_index++)
        if(cumulative_probs[cutoff_index] > top_p) break;
    
    std::vector<float> probs(cutoff_index+1);
    float sum_probs = 0.0f;
    for(int i=0; i<cutoff_index+1; i++){
        probs[i] = exp_logits[i];
        sum_probs += probs[i];
    }

    // normalize probabilities
    for(int i=0; i<cutoff_index+1; i++){
        probs[i] /= sum_probs;
    }

    // sample from the distribution
    std::random_device rd;
    std::mt19937 gen(rd());
    // std::vector<int> random_indices;
    // std::sample(probs.begin(), probs.end(), std::back_inserter(random_indices), 1, gen);
    std::discrete_distribution<int> dist(probs.begin(), probs.end());
    int sampled_index = dist(gen);
    int sampled_token = sorted_indices[sampled_index];
    
    return sampled_token;
}

void prefill(Arguments& args, Ort::Session& image_emb_session, Ort::Session& text_emb_session, 
    Ort::Session& decoding_session, Ort::AllocatorWithDefaultOptions& allocator, Ort::MemoryInfo& mem_info,
    NextInput* next_input, tokenizers::Tokenizer& tok) {

    // // create tokenizer obj
    // auto blob = LoadBytesFromFile(args.tokenizer_path);
    // auto tok = tokenizers::Tokenizer::FromBlobJSON(blob);

    // 1. image processing and embedding
    // process image
    std::pair<std::vector<float>, std::vector<int64_t>> image_data = ProcessImage(args.image_path);
    
    // create image tensor
    Ort::Value image_tensor = Ort::Value::CreateTensor<float>(
        mem_info, image_data.first.data(), image_data.first.size(),
        image_data.second.data(), image_data.second.size()
    );

    // create image input name
    InputOutputNames image_inout_names = GetInputOutputNames(image_emb_session, allocator);
    
    // run image embedding
    auto image_emb_output = image_emb_session.Run(Ort::RunOptions{nullptr},
        image_inout_names.input_names.data(), &image_tensor, 1,
        image_inout_names.output_names.data(), image_inout_names.output_names.size()
    );

    auto image_features_proj = image_emb_output[0].GetTensorMutableData<float>();
    auto image_features_proj_shape = image_emb_output[0].GetTensorTypeAndShapeInfo().GetShape();
    int64_t image_features_proj_len = image_features_proj_shape[1] * image_features_proj_shape[2];


    // 2. text embedding
    // prompt encode
    std::string prompt = args.input_text;
    std::vector<int> text_tok = tok.Encode(prompt);
    std::vector<int64_t> text_ids;
    text_ids.reserve(text_tok.size());

    // get image_token_pos and fill text_ids
    int64_t image_token_pos = 0;
    for(int id : text_tok){
        if(id == IMAGE_TOKEN_INDEX)  image_token_pos = text_ids.size();
        text_ids.push_back(static_cast<int64_t>(id));
    }

    std::vector<int64_t> text_shape = {BATCH_SIZE, static_cast<int64_t>(text_ids.size())};

    // create text tensor
    Ort::Value text_tensor = Ort::Value::CreateTensor<int64_t>(
        mem_info, text_ids.data(), text_ids.size(),
        text_shape.data(), text_shape.size()
    );

    // create text input name
    InputOutputNames text_inout_names = GetInputOutputNames(text_emb_session, allocator);

    // run text embedding
    auto text_emb_output = text_emb_session.Run(Ort::RunOptions{nullptr},
        text_inout_names.input_names.data(), &text_tensor, 1,
        text_inout_names.output_names.data(), text_inout_names.output_names.size()
    );


/**** half to float ****/
    // get tensor info
    Ort::TensorTypeAndShapeInfo text_emb_output_info = text_emb_output[0].GetTensorTypeAndShapeInfo();
    // element count
    int64_t text_emb_output_count = text_emb_output_info.GetElementCount();
    // get raw data
    // output is float16, but has no float16 format in onnxruntime
    // so we use uint16_t to store it
    uint16_t* text_emb_output_data = text_emb_output[0].GetTensorMutableData<uint16_t>();

    // convert to float
    std::vector<float> text_emb_output_float(text_emb_output_count);
    for (int64_t i = 0; i < text_emb_output_count; ++i) {
        text_emb_output_float[i] = Half2Float(text_emb_output_data[i]);
    }
/***********************/

    // auto input_features = text_emb_output[0].GetTensorMutableData<float>();
    auto input_features_shape = text_emb_output[0].GetTensorTypeAndShapeInfo().GetShape();
    int64_t input_features_len = input_features_shape[1] * input_features_shape[2];


    // 3. decoding
    // create hidden_states tensor
    std::vector<float> hidden_states(input_features_len + image_features_proj_len - 1);
    float* dst_ptr = hidden_states.data();
    int64_t hidden_dim = input_features_shape[2];
    int64_t input_offset = 0;

    std::memcpy(dst_ptr + input_offset, text_emb_output_float.data(), image_token_pos * hidden_dim * sizeof(float));
    input_offset += image_token_pos * hidden_dim;
    // std::cout << "image_token_pos : " << image_token_pos << std::endl;

    std::memcpy(dst_ptr + input_offset, image_features_proj, image_features_proj_shape[1]  * hidden_dim  * sizeof(float));
    input_offset += image_features_proj_shape[1] * hidden_dim;
    // std::cout << "image_features_proj_shape[1] : " << image_features_proj_shape[1] << std::endl;

    std::memcpy(dst_ptr + input_offset, text_emb_output_float.data() + (image_token_pos+1) * hidden_dim, 
                (input_features_shape[1] - image_token_pos - 1) * hidden_dim * sizeof(float));
    // std::cout << "input_features_shape[1] : " << input_features_shape[1] << std::endl;

    std::vector<int64_t> hidden_states_shape = {BATCH_SIZE, 
                    input_features_shape[1] + image_features_proj_shape[1] -1, 896};
    int64_t input_token_len = hidden_states_shape[1];
    // std::cout << "input_token_len : " << input_token_len << std::endl;

    Ort::Value hidden_states_tensor = Ort::Value::CreateTensor<float>(
        mem_info, hidden_states.data(), hidden_states.size(),
        hidden_states_shape.data(), hidden_states_shape.size()
    );

    // crate past_key_values tensor
    std::vector<Ort::Value> past_key_values;   
    std::vector<int64_t> past_key_value_shape = {BATCH_SIZE, KEY_VALUE_NUM, 0, 64};

    // create init past_key_values tensor
    for(int i=0; i<DECODER_NUM*2; i++){
        Ort::Value empty_key_value = Ort::Value::CreateTensor<float>(
            mem_info, nullptr, 0,
            past_key_value_shape.data(), past_key_value_shape.size()
        );
        past_key_values.push_back(std::move(empty_key_value));
    }

    // set init position_ids
    std::vector<int64_t> position_ids;
    std::vector<int64_t> position_ids_shape = {BATCH_SIZE, input_token_len};
    // position_ids.reserve(generated_ids.size());
    for(int64_t i=0; i<input_token_len; i++){
        position_ids.push_back(i);
    }

    // create init position_ids tensor
    Ort::Value position_ids_tensor = Ort::Value::CreateTensor<int64_t>(
        mem_info, position_ids.data(), position_ids.size(),
        position_ids_shape.data(), position_ids_shape.size()
    );

    // attention mask tensor
    std::vector<int64_t> attention_mask(input_token_len, 1);
    std::vector<int64_t> attention_mask_shape = {BATCH_SIZE, input_token_len};
    Ort::Value attention_mask_tensor = Ort::Value::CreateTensor<int64_t>(
        mem_info, attention_mask.data(), attention_mask.size(),
        attention_mask_shape.data(), attention_mask_shape.size()
    );

    // set prefill input tensors
    std::vector<Ort::Value> prefill_input_tensors;
    prefill_input_tensors.push_back(std::move(attention_mask_tensor));
    prefill_input_tensors.push_back(std::move(position_ids_tensor));
    for(int i=0; i<DECODER_NUM*2; i++){
        prefill_input_tensors.push_back(std::move(past_key_values[i]));
    }
    prefill_input_tensors.push_back(std::move(hidden_states_tensor));

    // create prefill inout name
    InputOutputNames decoding_inout_names = GetInputOutputNames(decoding_session, allocator);

    // run prefill
    auto prefill_output = decoding_session.Run(Ort::RunOptions{nullptr},
        decoding_inout_names.input_names.data(), prefill_input_tensors.data(),
        prefill_input_tensors.size(),
        decoding_inout_names.output_names.data(), decoding_inout_names.output_names.size()
    );

    // NextInput next_input;

    for(int i=0; i<DECODER_NUM*2; i++){
        next_input->past_key_values.push_back(std::move(prefill_output[i+1]));
    }

    std::vector<int64_t> next_token_shape = {BATCH_SIZE, 1};
    if(USE_SAMPLING){
        // sampling
        float* logits = prefill_output[0].GetTensorMutableData<float>();
        int64_t vocab_size = prefill_output[0].GetTensorTypeAndShapeInfo().GetShape()[2];
        int64_t last_token_index = input_token_len - 1;
        std::vector<float> logits_vector(logits + last_token_index * vocab_size,
                                    logits + (last_token_index+1) * vocab_size);
        int64_t next_token_id = top_p_sampling(logits_vector, TOP_P); //random sampling
        
        next_input->input_ids = Ort::Value::CreateTensor<int64_t>(
            mem_info, &next_token_id, 1,
            next_token_shape.data(), next_token_shape.size()
        );

        next_input->input_token_len = input_token_len;
    }else{
        // greedy
        float* logits = prefill_output[0].GetTensorMutableData<float>();
        int64_t vocab_size = prefill_output[0].GetTensorTypeAndShapeInfo().GetShape()[2];
        int64_t last_token_index = input_token_len - 1;
        int64_t next_token_id = Argmax(logits + last_token_index * vocab_size, vocab_size); // greedy sampling
        
        next_input->input_ids = Ort::Value::CreateTensor<int64_t>(
            mem_info, &next_token_id, 1,
            next_token_shape.data(), next_token_shape.size()
        );

        next_input->input_token_len = input_token_len;
    }

    next_input->generated_tokens.push_back(next_input->input_ids.GetTensorMutableData<int64_t>()[0]);

    // return next_input;
    return;
}

std::string decode(Arguments& args, Ort::Session& text_emb_session, 
    Ort::Session& decoding_session, Ort::AllocatorWithDefaultOptions& allocator, 
    Ort::MemoryInfo& mem_info, NextInput* first_input, tokenizers::Tokenizer& tok) {
    
    // Ort::Value next_token_tensor = std::move(first_input->input_ids);
    // std::cout << "[0]next token id : " << next_token_tensor.GetTensorMutableData<int64_t>()[0] << std::endl;
    // next token id
    std::vector<int64_t> next_token_shape = {BATCH_SIZE, 1};

    // position ids
    std::vector<int64_t> position_ids = {first_input->input_token_len};
    std::vector<int64_t> position_ids_shape = {BATCH_SIZE, 1};

    Ort::Value position_ids_tensor = Ort::Value::CreateTensor<int64_t>(
        mem_info, position_ids.data(), position_ids.size(),
        position_ids_shape.data(), position_ids_shape.size()
    );
    
    // past key values
    std::vector<Ort::Value> past_key_values;
    for (int i = 0; i < DECODER_NUM * 2; i++) {
        past_key_values.push_back(std::move(first_input->past_key_values[i]));
    }

    // attention mask
    std::vector<int64_t> attention_mask{1};
    std::vector<int64_t> attention_mask_shape = {BATCH_SIZE, 1};
    Ort::Value attention_mask_tensor = Ort::Value::CreateTensor<int64_t>(
        mem_info, attention_mask.data(), attention_mask.size(),
        attention_mask_shape.data(), attention_mask_shape.size()
    );

    
    // names setup
    InputOutputNames text_inout_names = GetInputOutputNames(text_emb_session, allocator);
    InputOutputNames decoding_inout_names = GetInputOutputNames(decoding_session, allocator);

    std::cout << "decoding..." << std::endl;

    std::cout << "gen token id : " << first_input->generated_tokens[0] << std::endl;
    // decoding loop
    for (int i = 0; i < MAX_NEW_TOKEN; i++) {

        // std::cout << "Step : " << i << std::endl;

        Ort::Value next_token_tensor = Ort::Value::CreateTensor<int64_t>(
            mem_info, &first_input->generated_tokens.back(), 1,
            next_token_shape.data(), next_token_shape.size()
        );


        // 1. hidden states
        auto embedding_output = text_emb_session.Run(Ort::RunOptions{nullptr},
            text_inout_names.input_names.data(), &next_token_tensor, 1,
            text_inout_names.output_names.data(), text_inout_names.output_names.size()
        );
        // std::cout << "embedding done" << std::endl;

    /**** half to float ****/
        // get tensor info
        Ort::TensorTypeAndShapeInfo text_emb_output_info = embedding_output[0].GetTensorTypeAndShapeInfo();
        // element count
        int64_t text_emb_output_count = text_emb_output_info.GetElementCount();
        // get raw data
        // output is float16, but has no float16 format in onnxruntime
        // so we use uint16_t to store it
        uint16_t* text_emb_output_data = embedding_output[0].GetTensorMutableData<uint16_t>();

        // convert to float
        std::vector<float> text_emb_output_float(text_emb_output_count);
        for (int64_t i = 0; i < text_emb_output_count; ++i) {
            text_emb_output_float[i] = Half2Float(text_emb_output_data[i]);
        }

        // create embedding output tensor
        Ort::Value embedding_output_tensor = Ort::Value::CreateTensor<float>(
            mem_info, text_emb_output_float.data(), text_emb_output_float.size(),
            text_emb_output_info.GetShape().data(), text_emb_output_info.GetDimensionsCount()
        );
    /***********************/

        // 2. decoding
        std::vector<Ort::Value> decoding_input_tensors;
        decoding_input_tensors.push_back(std::move(attention_mask_tensor));
        decoding_input_tensors.push_back(std::move(position_ids_tensor));
        for (int i = 0; i < DECODER_NUM * 2; i++) {
            decoding_input_tensors.push_back(std::move(past_key_values[i]));
        }
        decoding_input_tensors.push_back(std::move(embedding_output_tensor));
        
        // run decoding
        auto decoding_output = decoding_session.Run(Ort::RunOptions{nullptr},
            decoding_inout_names.input_names.data(), decoding_input_tensors.data(),
            decoding_input_tensors.size(),
            decoding_inout_names.output_names.data(), decoding_inout_names.output_names.size()
        );
        
        // 3. get next inputs setup
        // next token id
        if (USE_SAMPLING) {
            // sampling
            float* logits = decoding_output[0].GetTensorMutableData<float>();
            int64_t vocab_size = decoding_output[0].GetTensorTypeAndShapeInfo().GetShape()[2];
            int64_t last_token_index = position_ids[0] - 1;
            std::vector<float> logits_vector(logits + last_token_index * vocab_size,
                                             logits + (last_token_index + 1) * vocab_size);
            int64_t next_token_id = top_p_sampling(logits_vector, TOP_P); // random sampling
            
            next_token_tensor = Ort::Value::CreateTensor<int64_t>(
                mem_info, &next_token_id, 1,
                next_token_shape.data(), next_token_shape.size()
            );
            first_input->generated_tokens.push_back(next_token_id);
        } else {
            // greedy
            float* logits = decoding_output[0].GetTensorMutableData<float>();
            int64_t vocab_size = decoding_output[0].GetTensorTypeAndShapeInfo().GetShape()[2];
            int64_t last_token_index = position_ids[0] - 1;
            int64_t next_token_id = Argmax(logits + last_token_index * vocab_size, vocab_size); // greedy sampling

            next_token_tensor = Ort::Value::CreateTensor<int64_t>(
                mem_info, &next_token_id, 1,
                next_token_shape.data(), next_token_shape.size()
            );
            first_input->generated_tokens.push_back(next_token_id);
        }

        // check EOS
        if (first_input->generated_tokens.back() == EOS_TOKEN_ID) break;

        // position id
        position_ids[0]++;
        position_ids_tensor = Ort::Value::CreateTensor<int64_t>(
            mem_info, position_ids.data(), position_ids.size(),
            position_ids_shape.data(), position_ids_shape.size()
        );
        
        // past key values
        for (int i = 0; i < DECODER_NUM * 2; i++) {
            past_key_values[i] = std::move(decoding_output[i + 1]);
        }

        // attention mask --> ?? why??
        attention_mask_tensor = Ort::Value::CreateTensor<int64_t>(
            mem_info, attention_mask.data(), attention_mask.size(),
            attention_mask_shape.data(), attention_mask_shape.size()
        );

    }

    std::vector<int> generated_ids;
    generated_ids.reserve(first_input->generated_tokens.size());

    // convert int64_t to int
    for (int64_t token : first_input->generated_tokens) {
        generated_ids.push_back(static_cast<int>(token));
    }

    // decode tokens
    std::string decoded_text = tok.Decode(generated_ids);
    std::cout << "Decoded text: " << decoded_text << std::endl;

    return decoded_text;
}


InputOutputNames GetInputOutputNames(Ort::Session& session, Ort::AllocatorWithDefaultOptions& allocator){
    InputOutputNames names;
    size_t input_count = session.GetInputCount();
    size_t output_count = session.GetOutputCount();

    // get input names
    names.input_names_str.reserve(input_count);
    for(size_t i=0; i<input_count; i++){
        Ort::AllocatedStringPtr name = session.GetInputNameAllocated(i, allocator);
        names.input_names_str.push_back(name.get());
        names.input_names.push_back(names.input_names_str.back().c_str());
        // std::cout << "input[" << i << "] = " << name.get() << std::endl;
    }

    // get output names
    names.output_names_str.reserve(output_count);
    for(size_t i=0; i<output_count; i++){
        Ort::AllocatedStringPtr name = session.GetOutputNameAllocated(i, allocator);
        names.output_names_str.push_back(name.get());
        names.output_names.push_back(names.output_names_str.back().c_str());
        // std::cout << "output[" << i << "] = " << name.get() << std::endl;
    }
    
    return names;
}

float Half2Float(uint16_t value) {
    uint16_t h = value;
    uint32_t sign = (h & 0x8000) << 16;
    uint32_t exp = (h & 0x7C00) >> 10;
    uint32_t mant = (h & 0x03FF);

    uint32_t f;

    if (exp == 0) {
        if (mant == 0) {
            f = sign;
        } else {
            // Subnormal number
            while ((mant & 0x0400) == 0) {
                mant <<= 1;
                exp--;
            }
            exp++;
            mant &= ~0x0400;
            f = sign | ((exp + 112) << 23) | (mant << 13);
        }
    } else if (exp == 0x1F) {
        // Inf or NaN
        f = sign | 0x7F800000 | (mant << 13);
    } else {
        // Normalized number
        f = sign | ((exp + 112) << 23) | (mant << 13);
    }

    float result;
    std::memcpy(&result, &f, sizeof(result));
    return result;
}
