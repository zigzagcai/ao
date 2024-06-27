Search.setIndex({"docnames": ["api_ref_dtypes", "api_ref_intro", "api_ref_kernel", "api_ref_quantization", "api_ref_sparsity", "dtypes", "generated/torchao.dtypes.AffineQuantizedTensor", "generated/torchao.dtypes.to_affine_quantized", "generated/torchao.dtypes.to_nf4", "generated/torchao.quantization.Int4WeightOnlyGPTQQuantizer", "generated/torchao.quantization.Int4WeightOnlyQuantizer", "generated/torchao.quantization.SmoothFakeDynQuantMixin", "generated/torchao.quantization.SmoothFakeDynamicallyQuantizedLinear", "generated/torchao.quantization.int4_weight_only", "generated/torchao.quantization.int8_dynamic_activation_int4_weight", "generated/torchao.quantization.int8_dynamic_activation_int8_weight", "generated/torchao.quantization.int8_weight_only", "generated/torchao.quantization.quantize", "generated/torchao.quantization.smooth_fq_linear_to_inference", "generated/torchao.quantization.swap_linear_with_smooth_fq_linear", "generated/torchao.sparsity.PerChannelNormObserver", "generated/torchao.sparsity.WandaSparsifier", "generated/torchao.sparsity.apply_fake_sparsity", "generated/torchao.sparsity.apply_sparse_semi_structured", "getting-started", "index", "overview", "performant_kernels", "quantization", "sg_execution_times", "sparsity", "tutorials/index", "tutorials/sg_execution_times", "tutorials/template_tutorial"], "filenames": ["api_ref_dtypes.rst", "api_ref_intro.rst", "api_ref_kernel.rst", "api_ref_quantization.rst", "api_ref_sparsity.rst", "dtypes.rst", "generated/torchao.dtypes.AffineQuantizedTensor.rst", "generated/torchao.dtypes.to_affine_quantized.rst", "generated/torchao.dtypes.to_nf4.rst", "generated/torchao.quantization.Int4WeightOnlyGPTQQuantizer.rst", "generated/torchao.quantization.Int4WeightOnlyQuantizer.rst", "generated/torchao.quantization.SmoothFakeDynQuantMixin.rst", "generated/torchao.quantization.SmoothFakeDynamicallyQuantizedLinear.rst", "generated/torchao.quantization.int4_weight_only.rst", "generated/torchao.quantization.int8_dynamic_activation_int4_weight.rst", "generated/torchao.quantization.int8_dynamic_activation_int8_weight.rst", "generated/torchao.quantization.int8_weight_only.rst", "generated/torchao.quantization.quantize.rst", "generated/torchao.quantization.smooth_fq_linear_to_inference.rst", "generated/torchao.quantization.swap_linear_with_smooth_fq_linear.rst", "generated/torchao.sparsity.PerChannelNormObserver.rst", "generated/torchao.sparsity.WandaSparsifier.rst", "generated/torchao.sparsity.apply_fake_sparsity.rst", "generated/torchao.sparsity.apply_sparse_semi_structured.rst", "getting-started.rst", "index.rst", "overview.rst", "performant_kernels.rst", "quantization.rst", "sg_execution_times.rst", "sparsity.rst", "tutorials/index.rst", "tutorials/sg_execution_times.rst", "tutorials/template_tutorial.rst"], "titles": ["torchao.dtypes", "<code class=\"docutils literal notranslate\"><span class=\"pre\">torchao</span></code> API Reference", "torchao.kernel", "torchao.quantization", "torchao.sparsity", "Dtypes", "AffineQuantizedTensor", "to_affine_quantized", "to_nf4", "Int4WeightOnlyGPTQQuantizer", "Int4WeightOnlyQuantizer", "SmoothFakeDynQuantMixin", "SmoothFakeDynamicallyQuantizedLinear", "int4_weight_only", "int8_dynamic_activation_int4_weight", "int8_dynamic_activation_int8_weight", "int8_weight_only", "quantize", "smooth_fq_linear_to_inference", "swap_linear_with_smooth_fq_linear", "PerChannelNormObserver", "WandaSparsifier", "apply_fake_sparsity", "apply_sparse_semi_structured", "Getting Started", "Welcome to the torchao Documentation", "Overview", "Performant Kernels", "Quantization", "Computation times", "Sparsity", "&lt;no title&gt;", "Computation times", "Template Tutorial"], "terms": {"thi": [1, 6, 12, 14, 20, 21, 22, 33], "section": 1, "introduc": 1, "dive": 1, "detail": 1, "how": [1, 6], "integr": 1, "pytorch": [1, 25, 33], "optim": [1, 17], "your": [1, 17, 25], "machin": 1, "learn": [1, 33], "model": [1, 14, 17, 18, 19, 21, 22, 23, 25], "sparsiti": [1, 20, 21, 22, 23, 25], "quantiz": [1, 6, 9, 10, 11, 12, 13, 14, 15, 16, 18, 19, 25], "dtype": [1, 6, 7, 8, 17, 25], "kernel": [1, 6, 13, 17], "tba": [2, 5, 24, 26, 27, 28, 30], "class": [6, 9, 10, 11, 12, 20, 21], "torchao": [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23], "layout_tensor": 6, "aqtlayout": 6, "block_siz": [6, 7, 8], "tupl": [6, 7, 21], "int": [6, 7, 8, 10, 21], "shape": 6, "size": [6, 13, 14], "quant_min": [6, 7], "option": [6, 7, 10, 17, 18, 19, 21], "none": [6, 7, 17, 18, 19, 21], "quant_max": [6, 7], "zero_point_domain": [6, 7, 17], "zeropointdomain": [6, 7], "stride": 6, "sourc": [6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 25, 31, 33], "affin": [6, 17], "tensor": [6, 7, 8, 17, 21, 33], "subclass": [6, 12, 17, 20], "mean": 6, "we": [6, 17], "float": [6, 7, 17, 19, 21], "point": [6, 17], "an": [6, 21, 25], "transform": 6, "quantized_tensor": 6, "float_tensor": 6, "scale": [6, 11, 12, 18, 19], "zero_point": 6, "The": [6, 18, 19, 21], "repres": [6, 21], "look": 6, "extern": 6, "regardless": 6, "intern": 6, "represent": 6, "s": 6, "type": [6, 9, 10], "orient": 6, "field": 6, "serv": 6, "gener": [6, 31, 33], "layout": [6, 13], "storag": 6, "data": 6, "e": [6, 17], "g": [6, 17], "store": [6, 20], "plain": [6, 7], "int_data": 6, "pack": 6, "format": 6, "depend": 6, "devic": [6, 9, 10], "oper": 6, "granular": [6, 13, 14], "element": 6, "share": 6, "same": 6, "qparam": 6, "when": 6, "input": [6, 17, 21], "dimens": 6, "ar": [6, 13, 17, 21], "us": [6, 13, 14, 17, 21, 22, 25], "per": [6, 12, 13, 14, 15, 16, 21], "torch": [6, 12, 17, 18, 19, 22, 33], "minimum": 6, "valu": [6, 11, 12, 18, 21], "specifi": [6, 21], "deriv": 6, "from": [6, 14, 17, 29, 32, 33], "maximum": [6, 18], "domain": 6, "should": [6, 12, 20, 21], "eitehr": 6, "integ": 6, "zero": [6, 21], "ad": [6, 21], "dure": [6, 19], "subtract": 6, "unquant": 6, "default": [6, 17, 18, 19], "input_quant_func": 6, "callabl": [6, 17], "function": [6, 12, 17, 20, 21, 22, 25], "object": 6, "take": [6, 12, 17, 20], "output": [6, 33], "float32": 6, "dequant": 6, "given": 6, "return": [6, 17, 18, 19], "arg": [6, 11, 12, 21], "kwarg": [6, 11, 12, 20, 21, 23], "perform": [6, 11, 12, 18, 20], "convers": 6, "A": [6, 20], "infer": [6, 12, 18], "argument": [6, 17], "self": [6, 11, 12], "If": [6, 18, 21], "alreadi": 6, "ha": 6, "correct": 6, "otherwis": 6, "copi": [6, 21], "desir": 6, "here": 6, "wai": 6, "call": [6, 12, 17, 20], "non_block": 6, "fals": [6, 17, 18, 21], "memory_format": 6, "preserve_format": 6, "memori": 6, "tri": 6, "convert": [6, 12, 17], "asynchron": 6, "respect": 6, "host": 6, "possibl": 6, "cpu": 6, "pin": 6, "cuda": [6, 9, 10], "set": [6, 11, 12, 17, 18, 21], "new": [6, 17], "creat": 6, "even": 6, "match": 6, "other": [6, 21, 33], "exampl": [6, 17, 21, 29, 31, 32, 33], "randn": 6, "2": [6, 13, 17, 22, 33], "initi": 6, "float64": 6, "0": [6, 9, 11, 12, 17, 19, 21, 29, 32, 33], "5044": 6, "0005": 6, "3310": 6, "0584": 6, "cuda0": 6, "true": [6, 7, 9, 10, 17, 18], "input_float": 7, "mapping_typ": 7, "mappingtyp": 7, "target_dtyp": 7, "ep": 7, "scale_dtyp": 7, "zero_point_dtyp": [7, 17], "preserve_zero": [7, 17], "bool": [7, 10, 17, 18], "extended_layout": 7, "str": [7, 17, 19, 21], "inner_k_til": [7, 9, 10, 13], "64": [8, 9, 13], "scaler_block_s": 8, "256": [8, 10, 13], "blocksiz": 9, "128": [9, 13], "percdamp": 9, "01": 9, "groupsiz": [9, 10, 17], "8": [9, 10, 13], "padding_allow": [9, 10], "set_debug_x_absmax": [11, 12], "x_running_abs_max": [11, 12], "which": [11, 12], "lead": [11, 12], "smooth": [11, 12], "all": [11, 12, 20, 21, 22, 29, 31], "ones": [11, 12, 21], "alpha": [11, 12, 19], "5": [11, 12, 19, 21, 33], "enabl": [11, 12], "benchmark": [11, 12, 18], "without": [11, 12], "calibr": [11, 12], "replac": [12, 19], "nn": [12, 17, 18, 19], "linear": [12, 13, 14, 15, 16, 17, 19, 22], "implement": 12, "dynam": [12, 14, 15], "token": [12, 14, 15], "activ": [12, 14, 15, 18, 21], "channel": [12, 15, 16, 20], "weight": [12, 13, 14, 15, 16, 17, 21], "base": [12, 21], "smoothquant": [12, 18, 19], "forward": [12, 20], "x": [12, 17, 33], "defin": [12, 20, 21], "comput": [12, 20, 21], "everi": [12, 20], "overridden": [12, 20], "although": [12, 20], "recip": [12, 20], "pass": [12, 20], "need": [12, 20, 21], "within": [12, 20], "one": [12, 20], "modul": [12, 17, 18, 19, 20, 21], "instanc": [12, 17, 20], "afterward": [12, 20], "instead": [12, 20], "sinc": [12, 20], "former": [12, 20], "care": [12, 20], "run": [12, 17, 18, 20, 33], "regist": [12, 20], "hook": [12, 20], "while": [12, 20, 21], "latter": [12, 20], "silent": [12, 20], "ignor": [12, 20], "them": [12, 20], "classmethod": 12, "from_float": 12, "mod": 12, "fake": 12, "version": 12, "note": [12, 21], "requir": 12, "to_infer": 12, "calcul": [12, 18], "prepar": [12, 18, 21], "group_siz": [13, 14, 17], "appli": [13, 14, 15, 16, 17], "uint4": [13, 17], "onli": [13, 16, 17], "asymmetr": [13, 14, 17], "group": [13, 14], "layer": [13, 15, 16, 18, 19, 21, 22], "tensor_core_til": 13, "speedup": 13, "tinygemm": [13, 17], "paramet": [13, 14, 17, 18, 19, 21], "control": [13, 14, 21], "smaller": [13, 14], "more": [13, 14, 25], "fine": [13, 14], "grain": [13, 14], "choic": 13, "32": [13, 14, 17], "int4": [13, 14, 17], "mm": [13, 17], "4": [13, 22], "int8": [14, 15, 16, 17], "symmetr": [14, 15, 16], "produc": 14, "executorch": [14, 17], "backend": 14, "current": [14, 17, 19, 21], "did": 14, "support": 14, "lower": 14, "flow": [14, 22], "yet": 14, "apply_tensor_subclass": 17, "filter_fn": 17, "set_inductor_config": 17, "fulli": [17, 19], "qualifi": [17, 19], "name": [17, 19, 21], "want": 17, "whether": 17, "automat": [17, 33], "recommend": 17, "inductor": 17, "config": [17, 21], "import": [17, 33], "1": [17, 21, 29, 32, 33], "some": [17, 21], "predefin": 17, "method": [17, 21], "correspond": 17, "execut": [17, 29, 32], "path": 17, "also": 17, "customiz": 17, "int8_dynamic_activation_int4_weight": 17, "int8_dynamic_activation_int8_weight": 17, "op": 17, "compil": 17, "int4_weight_onli": 17, "int8_weight_onli": 17, "quant_api": 17, "m": 17, "sequenti": 17, "1024": 17, "write": 17, "own": 17, "you": [17, 21, 33], "can": 17, "add": [17, 33], "manual": 17, "constructor": 17, "to_affine_quant": 17, "groupwis": 17, "apply_weight_qu": 17, "lambda": 17, "int32": 17, "15": 17, "1e": 17, "6": 17, "bfloat16": 17, "under": [17, 25], "block0": 17, "submodul": 17, "def": 17, "fqn": [17, 21], "isinst": 17, "debug_skip_calibr": 18, "each": [18, 20], "smoothfakedynamicallyquantizedlinear": [18, 19], "contain": [18, 19], "debug": 18, "skip_fqn_list": 19, "cur_fqn": 19, "equival": 19, "list": [19, 21], "skip": [19, 21], "being": 19, "process": [19, 33], "factor": 19, "custom": 20, "observ": 20, "l2": 20, "norm": [20, 21], "buffer": 20, "x_orig": 20, "sparsity_level": 21, "semi_structured_block_s": 21, "wanda": 21, "sparsifi": 21, "prune": [21, 22, 25], "propos": 21, "http": 21, "arxiv": 21, "org": 21, "ab": 21, "2306": 21, "11695": 21, "awar": 21, "remov": 21, "product": 21, "magnitud": 21, "three": 21, "variabl": 21, "number": 21, "spars": 21, "block": 21, "out": 21, "target": 21, "level": 21, "dict": 21, "parametr": 21, "modifi": 21, "inplac": 21, "preserv": 21, "origin": 21, "deepcopi": 21, "squash_mask": 21, "params_to_keep": 21, "params_to_keep_per_lay": 21, "squash": 21, "mask": 21, "appropri": 21, "either": 21, "have": 21, "sparse_param": 21, "attach": 21, "kei": [21, 33], "save": 21, "param": 21, "specif": 21, "string": 21, "xdoctest": 21, "local": 21, "undefin": 21, "don": 21, "t": 21, "ani": 21, "hasattr": 21, "submodule1": 21, "keep": 21, "linear1": 21, "foo": 21, "bar": 21, "submodule2": 21, "linear42": 21, "baz": 21, "print": [21, 33], "42": 21, "24": 21, "update_mask": 21, "tensor_nam": 21, "statist": 21, "retriev": 21, "first": 21, "act_per_input": 21, "Then": 21, "metric": 21, "matrix": 21, "compar": 21, "across": 21, "whole": 21, "simul": 22, "It": 22, "ao": 22, "open": 25, "librari": 25, "provid": 25, "nativ": 25, "our": 25, "develop": 25, "content": 25, "come": 25, "soon": 25, "00": [29, 32], "003": [29, 32, 33], "total": [29, 32, 33], "file": [29, 32], "galleri": [29, 31, 33], "mem": [29, 32], "mb": [29, 32], "templat": [29, 31, 32], "tutori": [29, 31, 32], "tutorials_sourc": 29, "template_tutori": [29, 32, 33], "py": [29, 32, 33], "download": [31, 33], "python": [31, 33], "code": [31, 33], "tutorials_python": 31, "zip": 31, "jupyt": [31, 33], "notebook": [31, 33], "tutorials_jupyt": 31, "sphinx": [31, 33], "go": 33, "end": 33, "full": 33, "author": 33, "firstnam": 33, "lastnam": 33, "what": 33, "item": 33, "3": 33, "prerequisit": 33, "v2": 33, "gpu": 33, "describ": 33, "why": 33, "topic": 33, "link": 33, "relev": 33, "research": 33, "paper": 33, "walk": 33, "through": 33, "below": 33, "rand": 33, "2428": 33, "2850": 33, "0690": 33, "7516": 33, "6501": 33, "9909": 33, "2560": 33, "7202": 33, "1581": 33, "6603": 33, "3432": 33, "6239": 33, "7452": 33, "7700": 33, "8861": 33, "practic": 33, "user": 33, "test": 33, "knowledg": 33, "nlp": 33, "scratch": 33, "summar": 33, "concept": 33, "cover": 33, "highlight": 33, "takeawai": 33, "link1": 33, "link2": 33, "time": 33, "script": 33, "minut": 33, "second": 33, "ipynb": 33}, "objects": {"torchao.dtypes": [[6, 0, 1, "", "AffineQuantizedTensor"], [7, 2, 1, "", "to_affine_quantized"], [8, 2, 1, "", "to_nf4"]], "torchao.dtypes.AffineQuantizedTensor": [[6, 1, 1, "", "dequantize"], [6, 1, 1, "", "to"]], "torchao.quantization": [[9, 0, 1, "", "Int4WeightOnlyGPTQQuantizer"], [10, 0, 1, "", "Int4WeightOnlyQuantizer"], [11, 0, 1, "", "SmoothFakeDynQuantMixin"], [12, 0, 1, "", "SmoothFakeDynamicallyQuantizedLinear"], [13, 2, 1, "", "int4_weight_only"], [14, 2, 1, "", "int8_dynamic_activation_int4_weight"], [15, 2, 1, "", "int8_dynamic_activation_int8_weight"], [16, 2, 1, "", "int8_weight_only"], [17, 2, 1, "", "quantize"], [18, 2, 1, "", "smooth_fq_linear_to_inference"], [19, 2, 1, "", "swap_linear_with_smooth_fq_linear"]], "torchao.quantization.SmoothFakeDynQuantMixin": [[11, 1, 1, "", "set_debug_x_absmax"]], "torchao.quantization.SmoothFakeDynamicallyQuantizedLinear": [[12, 1, 1, "", "forward"], [12, 1, 1, "", "from_float"], [12, 1, 1, "", "set_debug_x_absmax"], [12, 1, 1, "", "to_inference"]], "torchao": [[4, 3, 0, "-", "sparsity"]], "torchao.sparsity": [[20, 0, 1, "", "PerChannelNormObserver"], [21, 0, 1, "", "WandaSparsifier"], [22, 2, 1, "", "apply_fake_sparsity"], [23, 2, 1, "", "apply_sparse_semi_structured"]], "torchao.sparsity.PerChannelNormObserver": [[20, 1, 1, "", "forward"]], "torchao.sparsity.WandaSparsifier": [[21, 1, 1, "", "prepare"], [21, 1, 1, "", "squash_mask"], [21, 1, 1, "", "update_mask"]]}, "objtypes": {"0": "py:class", "1": "py:method", "2": "py:function", "3": "py:module"}, "objnames": {"0": ["py", "class", "Python class"], "1": ["py", "method", "Python method"], "2": ["py", "function", "Python function"], "3": ["py", "module", "Python module"]}, "titleterms": {"torchao": [0, 1, 2, 3, 4, 25], "dtype": [0, 5], "api": [1, 25], "refer": [1, 25], "python": 1, "kernel": [2, 27], "quantiz": [3, 17, 28], "sparsiti": [4, 30], "affinequantizedtensor": 6, "to_affine_quant": 7, "to_nf4": 8, "int4weightonlygptqquant": 9, "int4weightonlyquant": 10, "smoothfakedynquantmixin": 11, "smoothfakedynamicallyquantizedlinear": 12, "int4_weight_onli": 13, "int8_dynamic_activation_int4_weight": 14, "int8_dynamic_activation_int8_weight": 15, "int8_weight_onli": 16, "smooth_fq_linear_to_infer": 18, "swap_linear_with_smooth_fq_linear": 19, "perchannelnormobserv": 20, "wandasparsifi": 21, "apply_fake_spars": 22, "apply_sparse_semi_structur": 23, "get": 24, "start": 24, "welcom": 25, "document": 25, "overview": [26, 33], "perform": 27, "comput": [29, 32], "time": [29, 32], "templat": 33, "tutori": 33, "step": 33, "option": 33, "addit": 33, "exercis": 33, "conclus": 33, "further": 33, "read": 33}, "envversion": {"sphinx.domains.c": 2, "sphinx.domains.changeset": 1, "sphinx.domains.citation": 1, "sphinx.domains.cpp": 6, "sphinx.domains.index": 1, "sphinx.domains.javascript": 2, "sphinx.domains.math": 2, "sphinx.domains.python": 3, "sphinx.domains.rst": 2, "sphinx.domains.std": 2, "sphinx.ext.intersphinx": 1, "sphinx.ext.todo": 2, "sphinx.ext.viewcode": 1, "sphinx": 56}})