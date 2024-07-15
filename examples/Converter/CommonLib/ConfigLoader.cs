using Newtonsoft.Json;
using System.Collections.Generic;
using System.IO;


namespace Converter.CommonLib
{
	internal class ConfigLoader
    {
        /// <summary>
        /// general_architecture && tokenizer_ggml_pre
        /// </summary>
        public string model_type { get; set; }

        /// <summary>
        /// general_name 
        /// </summary>
        public string general_name { get; set; }

        /// <summary>
        /// model_block_count
        /// </summary>
        public uint num_hidden_layers { get; set; }

        /// <summary>
        /// 
        /// model_context_length 
        /// </summary>
        public uint max_position_embeddings { get; set; }

        /// <summary>
        /// model_embedding_length
        /// </summary>
        public uint hidden_size { get; set; }

        /// <summary>
        /// model_feed_forward_length
        /// </summary>
        public uint intermediate_size { get; set; }

        /// <summary>
        /// model_attention_head_count
        /// </summary>
        public uint num_attention_heads { get; set; }

        /// <summary>
        /// model_attention_head_count_kv
        /// </summary>
        public uint num_key_value_heads { get; set; }

        /// <summary>
        /// model_rope_freq_base
        /// </summary>
        public float rope_theta { get; set; }

        /// <summary>
        /// model_attention_layer_norm_rms_epsilon
        /// </summary>
        public float rms_norm_eps { get; set; }

        /// <summary>
        /// general_file_type
        /// </summary>
        public uint general_file_type { get; set; }

        /// <summary>
        /// tokenizer_ggml_model
        /// </summary>
        public string tokenizer_ggml_model { get; set; }

        /// <summary>
        /// tokenizer.ggml.tokens
        /// </summary>
        public List<string> tokenizer_ggml_tokens { get; set; } = new List<string>();

        /// <summary>
        /// tokenizer.ggml.token_type
        /// </summary>
        public List<int> tokenizer_ggml_token_type { get; set; } = new List<int>();

        /// <summary>
        /// tokenizer.ggml.merges
        /// </summary>
        public List<string> tokenizer_ggml_merges { get; set; } = new List<string>();

        /// <summary>
        /// tokenizer.ggml.eos_token_id
        /// </summary>
        public uint eos_token_id { get; set; }

        /// <summary>
        /// tokenizer.ggml.padding_token_id
        /// in generation_config.json
        /// </summary>
        public uint pad_token_id { get; set; }

        /// <summary>
        /// tokenizer.ggml.bos_token_id
        /// </summary>
        public uint bos_token_id { get; set; }

        /// <summary>
        /// tokenizer.chat_template
        /// </summary>
        public string chat_template { get; set; }
        public uint vocab_size { get; set; }
        public bool add_bos_token { get; set; }
        public bool add_eos_token { get; set; }

        internal void LoadFromFolder(string folderPath)
        {
            //safetensors_stroge_type safetensors_Stroge_Type = safetensors_stroge_type.none;

            if (!File.Exists(Path.Combine(folderPath, "config.json")))
            {
                throw new FileNotFoundException("config.json not found in the specified folder");
            }
            if (!File.Exists(Path.Combine(folderPath, "generation_config.json")))
            {
                throw new FileNotFoundException("generation_config.json not found in the specified folder");
            }
            if (!File.Exists(Path.Combine(folderPath, "tokenizer_config.json")))
            {
                throw new FileNotFoundException("tokenizer_config.json not found in the specified folder");
            }

            ConfigLoader configLoader = JsonConvert.DeserializeObject<ConfigLoader>(File.ReadAllText(Path.Combine(folderPath, "config.json")));
            generation_config generationConfigLoader = JsonConvert.DeserializeObject<generation_config>(File.ReadAllText(Path.Combine(folderPath, "generation_config.json")));
            tokenizer_config tokenizerConfig = JsonConvert.DeserializeObject<tokenizer_config>(File.ReadAllText(Path.Combine(folderPath, "tokenizer_config.json")));

            if (File.Exists(Path.Combine(folderPath, "tokenizer.json")))
            {
                tokenizer tokenizerLoader = JsonConvert.DeserializeObject<tokenizer>(File.ReadAllText(Path.Combine(folderPath, "tokenizer.json")));

                foreach (var a in tokenizerLoader.model.vocab)
                {
                    //tokenizer_ggml_tokens.Add(a.Key);
                    tokenizer_ggml_tokens.Add(a.Key);
                }
                foreach (var a in tokenizerLoader.added_tokens)
                {
                    //tokenizer_ggml_tokens.Add(a.content);
                    tokenizer_ggml_tokens.Add(a.content);
                }
                foreach (var a in tokenizerLoader.model.merges)
                {
                    //tokenizer_ggml_merges.Add(a);
                    tokenizer_ggml_merges.Add(a);
                }
                for (int i = 0; i < configLoader.vocab_size; i++)
                {
                    tokenizer_ggml_token_type.Add(1);
                }

                for (int i = tokenizer_ggml_tokens.Count; i < (int)configLoader.vocab_size; i++)
                {
                    tokenizer_ggml_tokens.Add($"[PAD{i}]");
                }

            }

            intermediate_size = configLoader.intermediate_size;
            hidden_size = configLoader.hidden_size;
            rms_norm_eps = configLoader.rms_norm_eps;
            max_position_embeddings = configLoader.max_position_embeddings;
            num_attention_heads = configLoader.num_attention_heads;
            num_hidden_layers = configLoader.num_hidden_layers;
            num_key_value_heads = configLoader.num_key_value_heads;
            rope_theta = configLoader.rope_theta;
            tokenizer_ggml_model = configLoader.tokenizer_ggml_model;
            pad_token_id = generationConfigLoader.pad_token_id;
            chat_template = tokenizerConfig.chat_template;
            vocab_size = configLoader.vocab_size;
            eos_token_id = configLoader.eos_token_id;
            bos_token_id = configLoader.bos_token_id;
            model_type = configLoader.model_type;
            add_bos_token = tokenizerConfig.add_bos_token;
            add_eos_token = tokenizerConfig.add_eos_token;
        }

        private class generation_config
        {
            public uint pad_token_id { get; set; }
        }

        private class tokenizer_config
        {
            public bool add_bos_token { get; set; }
            public bool add_eos_token { get; set; }

            public string chat_template { get; set; }
            public class added_tokens_decoder_class
            {
                public Dictionary<int, detail> added_tokens_decoder { get; set; }
                public class detail
                {
                    public string content { get; set; }
                }
            }
        }
        private class tokenizer
        {
            public List<added_tokens_class> added_tokens { get; set; }
            public model_class model { get; set; }
            public class added_tokens_class
            {
                public uint id { get; set; }
                public string content { get; set; }
            }
            public class model_class
            {
                public string type { get; set; }
                public Dictionary<string, int> vocab { get; set; }
                public List<string> merges { get; set; }
            }

        }
    }
}
