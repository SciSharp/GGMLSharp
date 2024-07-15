using ProtoBuf;
using System.Collections.Generic;


namespace Converter.CommonLib
{
	[ProtoContract]
    internal class Sentencepiece
    {
        [ProtoMember(1)]
        // Sentence pieces with scores.
        public List<SentencePiece> pieces { get; set; }

        [ProtoMember(2)]
        // Spec used to generate this model file.
        public TrainerSpec trainer_spec { get; set; }

        [ProtoMember(3)]
        // Spec for text normalization.
        public NormalizerSpec normalizer_spec { get; set; }

        [ProtoMember(4)]
        // Stores sample input and its expected segmentation to verify the model.
        public SelfTestData self_test_data { get; set; }

        [ProtoMember(5)]
        // Spec for text de-normalization.
        public NormalizerSpec denormalizer_spec { get; set; }

        [ProtoContract]
        public class SentencePiece
        {
            public enum Type
            {
                NORMAL = 1,        // normal symbol
                UNKNOWN = 2,       // unknown symbol. only <unk> for now.
                CONTROL = 3,       // control symbols. </s>, <s>, <2ja> etc.
                USER_DEFINED = 4,  // user defined symbols.
                                   // Typical usage of USER_DEFINED symbol
                                   // is placeholder.
                BYTE = 6,         // byte symbols. Used when `byte_fallback` is true.
                UNUSED = 5,        // this piece is not used.
            }
            [ProtoMember(1)]
            public string piece { get; set; }

            [ProtoMember(2)]
            // piece must not be empty.
            public float score { get; set; }

            [ProtoMember(3)]
            public Type type { get; set; } = Type.NORMAL;

        }

        [ProtoContract]
        public class TrainerSpec
        {
            ///////////////////////////////////////////////////////////////////
            // General parameters
            //
            // Input corpus files.
            //  Trainer accepts the following two formats:
            //  A) Monolingual: plain text, one sentence per line.
            //  B) Bilingual:   TSV, source sentence <tab> target sentence
            //  When bilingual data is passed, shared vocabulary model is built.
            //  Note that the input file must be raw corpus, not a preprocessed corpus.
            //  Trainer only loads the first `input_sentence_size` sentences specified
            //  with this parameter.
            [ProtoMember(1)]
            public List<string> input { get; set; }

            // Input corpus format:
            // "text": one-sentence-per-line text format (default)
            // "tsv":  sentence <tab> freq
            [ProtoMember(7)]
            public string input_format { get; set; }

            // Output model file prefix.
            // <model_prefix>.model and <model_prefix>.vocab are generated.
            [ProtoMember(2)]
            public string model_prefix { get; set; }

            // Model Type. only have UNIGRAM now.
            public enum ModelType
            {
                UNIGRAM = 1,  // Unigram language model with dynamic algorithm
                BPE = 2,      // Byte Pair Encoding
                WORD = 3,     // Delimitered by whitespace.
                CHAR = 4,     // tokenizes into character sequence
            }
            [ProtoMember(3)]
            public ModelType model_type { get; set; } = ModelType.UNIGRAM;

            // Vocabulary size. 8k is the default size.
            [ProtoMember(4)]
            public int vocab_size { get; set; } = 8000;

            // List of the languages this model can accept.
            // Since the model is language-agnostic, this field is used as a reference.
            [ProtoMember(5)]
            public List<string> accept_language { get; set; }

            // Size of self-test samples, which are encoded in the model file.
            [ProtoMember(6)]
            public int self_test_sample_size { get; set; } = 0;

            // Whether to use DP version of sentencepiece. Use it with TSV input format
            // (requires precomputed word tab counts to work).
            [ProtoMember(50)]
            public bool enable_differential_privacy { get; set; } = false;

            // Set these parameters if you need DP version of sentencepiece.
            // std of noise to add.
            [ProtoMember(51)]
            public float differential_privacy_noise_level { get; set; } = 0;

            // Clipping threshold to apply after adding noise. All the words with
            // frequency less than this value are dropped.
            [ProtoMember(52)]
            public ulong differential_privacy_clipping_threshold { get; set; } = 0;

            ///////////////////////////////////////////////////////////////////
            // Training parameters.
            //
            // Uses characters which cover the corpus with the ratio of `chars_coverage`.
            // This parameter determines the set of basic Alphabet of sentence piece.
            // 1.0 - `chars_coverage` characters are treated as UNK.
            // See also required_chars field.
            [ProtoMember(10)]
            public float character_coverage { get; set; } = 0.9995f;

            // Maximum size of sentences the trainer loads from `input` parameter.
            // Trainer simply loads the `input` files in sequence.
            // It is better to shuffle the input corpus randomly.
            [ProtoMember(11)]
            public ulong input_sentence_size { get; set; } = 0;

            [ProtoMember(19)]
            public bool shuffle_input_sentence { get; set; } = false;

            // Maximum size of sentences to make seed sentence pieces.
            // Extended suffix array is constructed to extract frequent
            // sub-strings from the corpus. This uses 20N working space,
            // where N is the size of corpus.
            [ProtoMember(12)]
            public int mining_sentence_size { get; set; } = 0;

            // Maximum size of sentences to train sentence pieces.
            [ProtoMember(13)]
            public int training_sentence_size { get; set; } = 0;

            // The size of seed sentencepieces.
            // `seed_sentencepiece_size` must be larger than `vocab_size`.
            [ProtoMember(14)]
            public int seed_sentencepiece_size { get; set; } = 1000000;

            // In every EM sub-iterations, keeps top
            // `shrinking_factor` * `current sentencepieces size` with respect to
            // the loss of the sentence piece. This value should be smaller than 1.0.
            [ProtoMember(15)]
            public float shrinking_factor { get; set; } = 0.75f;

            // The maximum sentence length in byte. The sentences with the length
            // larger than `max_sentence_length` is simply ignored.
            // Longer input tends to bring the following risks:
            //  * Overflow during EM training (unigram language model only)
            //  * Performance drop because of O(n log n) cost in BPE.
            [ProtoMember(18)]
            public int max_sentence_length { get; set; } = 4192;

            // Number of threads in the training.
            [ProtoMember(16)]
            public int num_threads { get; set; } = 16;

            // Number of EM sub iterations.
            [ProtoMember(17)]
            public int num_sub_iterations { get; set; } = 2;

            ///////////////////////////////////////////////////////////////////
            // SentencePiece parameters which control the shapes of sentence piece.
            //
            // Maximum length of sentencepiece.
            [ProtoMember(20)]
            public int max_sentencepiece_length { get; set; } = 16;

            // Uses Unicode script to split sentence pieces.
            // When `split_by_unicode_script` is true, we do not allow sentence piece to
            // include multiple Unicode scripts, e.g. "F1" is not a valid piece.
            // Exception: CJ characters (Hiragana/Katakana/Han) are all handled
            // as one script Type, since Japanese word can consist of multiple scripts.
            // This exception is always applied regardless of the accept-language
            // parameter.
            [ProtoMember(21)]
            public bool split_by_unicode_script { get; set; } = true;

            // When `split_by_number` is true, put a boundary between number and
            // non-number transition. If we want to treat "F1" is one token, set this flag
            // to be false.
            [ProtoMember(23)]
            public bool split_by_number { get; set; } = true;

            // Use a white space to split sentence pieces.
            // When `split_by_whitespace` is false, we may have the piece containing
            // a white space in the middle. e.g., "in_the".
            [ProtoMember(22)]
            public bool split_by_whitespace { get; set; } = true;

            // Adds whitespace symbol (_) as a suffix instead of prefix. e.g., _hello =>
            // hello_. When `treat_whitespace_as_suffix` is true,
            // NormalizerSpec::add_dummy_prefix will add the dummy whitespace to the end
            // of sentence.
            [ProtoMember(24)]
            public bool treat_whitespace_as_suffix { get; set; } = false;

            // Allows pieces that only contain whitespaces instead of appearing only as
            // prefix or suffix of other pieces.
            [ProtoMember(26)]
            public bool allow_whitespace_only_pieces { get; set; } = false;

            // Split all digits (0-9) into separate pieces.
            [ProtoMember(25)]
            public bool split_digits { get; set; } = false;

            // Defines the pre-tokenization delimiter.
            // When specified, no pieces crossing this delimiter is not included
            // in the vocab. Then the delimiter string is virtually ignored
            // during the training. This field can allows constraints on the vocabulary
            // selection. Note that this field is available on unigram mode.
            [ProtoMember(53)]
            public string pretokenization_delimiter { get; set; } = string.Empty;

            ///////////////////////////////////////////////////////////////////
            // Vocabulary management
            //
            // Defines control symbols used as an indicator to
            // change the behavior of the decoder. <s> and </s> are pre-defined.
            // We can use this field to encode various meta information,
            // including language indicator in multilingual model.
            // These symbols are not visible to users, but visible to
            // the decoder. Note that when the input sentence contains control symbols,
            // they are not treated as one token, but segmented into normal pieces.
            // Control symbols must be inserted independently from the segmentation.
            [ProtoMember(30)]
            public List<string> control_symbols { get; set; }

            // Defines user defined symbols.
            // These symbols are added with extremely high score
            // so they are always treated as one unique symbol in any context.
            // Typical usage of user_defined_symbols is placeholder for named entities.
            [ProtoMember(31)]
            public List<string> user_defined_symbols { get; set; }

            // Defines required characters. Each UTF8 character in this string is included
            // in the character set regardless of character_coverage value. Unlike
            // user_defined_symbols, these characters have scores based on the frequency
            // on input sentences, and the model can form subwords using characters
            // in this field.
            [ProtoMember(36)]
            public string required_chars { get; set; }

            // Decomposes unknown pieces into UTF-8 bytes.
            [ProtoMember(35)]
            public bool byte_fallback { get; set; } = false;

            // When creating the vocabulary file, defines whether or not to additionally
            // output the score for each piece.
            [ProtoMember(32)]
            public bool vocabulary_output_piece_score { get; set; } = true;

            // `vocab_size` is treated as hard limit. Crash if
            // the model can not produce the vocab of size `vocab_size`,
            // When `hard_vocab_limit` is false, vocab_size is treated
            // as soft limit. Note that when model_type=char,
            // always assumes hard_vocab_limit = false.
            [ProtoMember(33)]
            public bool hard_vocab_limit { get; set; } = true;

            // use all symbols for vocab extraction. This flag is valid
            // if model Type is either CHAR or WORD
            [ProtoMember(34)]
            public bool use_all_vocab { get; set; } = false;

            ///////////////////////////////////////////////////////////////////
            // Reserved special meta tokens.
            // * -1 is not used.
            // * unk_id must not be -1.
            // Id must starts with 0 and be contigous.
            [ProtoMember(40)]
            public int unk_id { get; set; } = 0;   // <unk>
            [ProtoMember(41)]
            public int bos_id { get; set; } = 1;   // <s>

            [ProtoMember(42)]
            public int eos_id { get; set; } = 2;   // </s>

            [ProtoMember(43)]
            public int pad_id { get; set; } = -1;  // <pad> (padding)

            [ProtoMember(45)]
            public string unk_piece { get; set; } = "<unk>";

            [ProtoMember(46)]
            public string bos_piece { get; set; } = "<s>";

            [ProtoMember(47)]
            public string eos_piece { get; set; } = "</s>";

            [ProtoMember(48)]
            public string pad_piece { get; set; } = "<pad>";

            // Encodes <unk> into U+2047 (DOUBLE QUESTION MARK),
            // since this character can be useful both for user and
            // developer. We can easily figure out that <unk> is emitted.
            [ProtoMember(44)]
            public string unk_surface { get; set; } = " \xE2\x81\x87 ";

            // Increase bit depth to allow unigram model training on large
            // (>10M sentences) corpora. A Side-effect of enabling this flag
            // is increased memory usage.
            [ProtoMember(49)]
            public bool train_extremely_large_corpus { get; set; } = false;

            // Path to a seed sentencepieces file, with one tab-separated
            // seed sentencepiece <tab> frequency per line.
            [ProtoMember(54)]
            public string seed_sentencepieces_file { get; set; } = string.Empty;
        }

        [ProtoContract]
        public class NormalizerSpec
        {
            [ProtoMember(1)]
            // name of normalization rule.
            public string name { get; set; }

            // Pre-compiled normalization rule created by
            // Builder::GetPrecompiledCharsMap() or Builder::CompileCharsMap() method.
            // Usually this field is set by Builder::GetNormalizerSpec() method.
            [ProtoMember(2)]
            public List<byte> precompiled_charsmap { get; set; }


            // Adds dummy whitespace at the beginning of text in order to
            // treat "world" in "world" and "hello world" in the same way.
            [ProtoMember(3)]
            public bool add_dummy_prefix { get; set; } = true;

            // Removes leading, trailing, and duplicate internal whitespace.
            [ProtoMember(4)]
            public bool remove_extra_whitespaces { get; set; } = true;


            // Replaces whitespace with meta symbol.
            // This field must be true to train sentence piece model.
            [ProtoMember(5)]
            public bool escape_whitespaces { get; set; } = true;


            // Custom normalization rule file in TSV format.
            // https://github.com/google/sentencepiece/blob/master/doc/normalization.md
            // This field is only used in SentencePieceTrainer::Train() method, which
            // compiles the rule into the binary rule stored in `precompiled_charsmap`.
            [ProtoMember(6)]
            public string normalization_rule_tsv { get; set; }

        }

        [ProtoContract]
        public class SelfTestData
        {
            [ProtoContract]
            public class Sample
            {
                [ProtoMember(1)]
                public string input;
                [ProtoMember(2)]
                public string expected;
            }
            [ProtoMember(1)]
            public List<Sample> samples;
        }

    }
}
