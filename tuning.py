import subprocess
import csv
import ast
import math
import random
import re
import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV
from skopt import gp_minimize
from skopt.space import Integer, Real, Categorical
from skopt.utils import use_named_args
from typing import List, Union, Dict

class CephParameter:
    def __init__(self, name: str, description: str, param_type: str,
                 default_value: Union[str, float, int, bool], constraint: Union[str, List[Union[str, int, float]]]):
        self.name = name
        self.description = description
        self.param_type = param_type
        self.default_value = self._convert_value(default_value, param_type)
        self.constraint = self._parse_constraint(constraint, param_type)
        self.one_hot_encoding = self.process_categorical(self.default_value) if param_type == "str" and self.constraint != "dynamic" else None
        self.normalized_value = self.normalize_value() if param_type in ["float", "int", "uint", "size"] else self.default_value

    def _convert_value(self, value: Union[str, float, int, bool], param_type: str):
        if param_type == "str":
            return str(value).strip()
        elif param_type == "float":
            return float(value)
        elif param_type == "int":
            return int(value)
        elif param_type == "bool":
            return value if isinstance(value, bool) else value.lower() == "true"
        elif param_type == "size":
            default_value, size_unit = self._parse_size(value)
            self.size_unit = size_unit 
            return default_value
        else:
            raise ValueError(f"Unsupported parameter type: {param_type}")
    
    def _parse_size(self, size_str: str):
        size_units = {"K": 1024, "M": 1024**2, "G": 1024**3, "T": 1024**4}

        for unit in size_units:
            if size_str.endswith(unit + "i"):
                return int(size_str[:-2]), unit
        
        raise ValueError(f"Invalid size format: {size_str}")
    
    def _parse_constraint(self, constraint: str, param_type: str):        
        try:
            parsed_constraint = ast.literal_eval(constraint)
            if isinstance(parsed_constraint, list):
                if param_type == "int" and len(parsed_constraint) == 2:
                    return (parsed_constraint[0], parsed_constraint[1])
                elif param_type == "float" and len(parsed_constraint) == 2:
                    return (parsed_constraint[0], parsed_constraint[1])
                return parsed_constraint
        except (ValueError, SyntaxError):
            pass
        
        if param_type == "str":
            return [item.strip() for item in constraint.split(",")]
                
        return constraint

    def process_categorical(self, value: str) -> Dict[str, int]:
        """Convert categorical variables into one-hot encoding."""
        one_hot_dict = {}
        if isinstance(self.constraint, list):
            for category in self.constraint:
                key = f"{category.strip()}"
                one_hot_dict[key] = int(value.strip() == category.strip())
        return one_hot_dict

    def normalize_value(self):
        """Apply log1p transformation for numeric values."""
        if isinstance(self.default_value, (int, float)) and self.default_value >= 0:
            return math.log1p(self.default_value)
        return self.default_value
    
    def __repr__(self):
        return (f"CephParameter(name={self.name}, param_type={self.param_type}, "
                f"default_value={self.default_value}, normalized_value={self.normalized_value}, "
                f"one_hot_encoding={self.one_hot_encoding})")

class CephParameterParser:
    @staticmethod
    def parse_csv(file_path: str) -> List[CephParameter]:
        parameters = []
        with open(file_path, newline='', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            next(reader)  # Skip header row
            for row in reader:
                name, description, param_type, default_value, constraint = row
                parameters.append(CephParameter(name, description, param_type, default_value, constraint))
        return parameters

def sample_configurations(parameters: List[CephParameter], n: int) -> List[Dict[str, Union[str, int, float]]]:
    sampled_configs = []
    size_units = {"K": 1024, "M": 1024**2, "G": 1024**3, "T": 1024**4}

    # 向sampled_config中添加默认配置
    default_config = {}
    for param in parameters:
        if param.param_type == "size":
            default_config[param.name] = param.default_value * size_units[param.size_unit]
        else:
            default_config[param.name] = param.default_value
    sampled_configs.append(default_config)

    for _ in range(n):
        config = {}
        for param in parameters:
            # 如果参数是布尔类型，那么随机选择True或False
            if param.param_type == "bool":
                config[param.name] = random.choice([True, False])
            # 如果参数是字符串类型，那么随机选择一个值
            elif param.param_type == "str":
                config[param.name] = random.choice(param.constraint)
            # 如果参数是整数或者浮点数类型且存在限定范围，那么在限定范围内随机取值
            elif isinstance(param.constraint, tuple) or isinstance(param.constraint, list):
                lower = param.constraint[0]
                upper = param.constraint[1]
                if param.param_type in ["int", "size"]:
                    config_val = random.randint(lower, upper)
                else:
                    config_val = random.uniform(lower, upper)
                if param.param_type == "size":
                    config_val = config_val * size_units[param.size_unit]
                config[param.name] = config_val
        sampled_configs.append(config)
    return sampled_configs

def normalize_sampled_config(config: Dict[str, Union[str, int, float]], params: List[CephParameter]) -> Dict[str, Union[str, int, float]]:
    normalized_config = {}
    for index, (param_name, param_value) in enumerate(config.items()):
        if isinstance(param_value, str):
            # 只处理类似"145M"这样的字符串（数字+单个字母），将M去掉然后保留145，并转化成整数然后math.log1p
            if param_value[-1].isalpha() and param_value[:-1].isdigit():
                param_value = int(param_value[:-1])
                if params[index].default_value < 100:
                    normalized_config[param_name] = param_value
                else:
                    normalized_config[param_name] = math.log1p(param_value)
            else:
                for param in params:
                    if param.name == param_name:
                        normalized_config[param_name] = param.process_categorical(param_value)
                        break
        elif isinstance(param_value, (int, float)) and param_value >= 0:
            if params[index].default_value < 100:
                normalized_config[param_name] = param_value
            else:
                normalized_config[param_name] = math.log1p(param_value)
        elif isinstance(param_value, bool):
            normalized_config[param_name] = int(param_value)
    return normalized_config

def flatten_configuration(config: Dict[str, Union[Dict, float, int]]) -> Dict[str, float]:
    """
    展平嵌套的配置字典，将one-hot编码的字段转换为单独的键值对。
    """
    flat_config = {}
    for key, value in config.items():
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                flat_config[f"{key}_{sub_key}"] = float(sub_value)
        else:
            flat_config[key] = float(value)
    return flat_config

def extract_important_features(
    configurations: List[Dict[str, Union[Dict, float, int]]], 
    performance_metrics: List[float], 
    alpha_values: List[float] = None
) -> List[str]:
    """
    使用 Lasso 进行特征选择，找出对性能影响最大的 Ceph 配置参数
    """
    flattened_configs = [flatten_configuration(config) for config in configurations]
    df = pd.DataFrame(flattened_configs)
    df.fillna(0, inplace=True)

    y = np.array(performance_metrics)
    if alpha_values is None:
        alpha_values = np.logspace(-4, 0, 50)

    lasso = LassoCV(alphas=alpha_values, cv=2, random_state=42, max_iter=10000)
    lasso.fit(df, y)
    
    important_features = df.columns[lasso.coef_ != 0].tolist()
    return important_features

class CephBayesianOptimizer:
    def __init__(self, parameters, func):
        size_units = {"K": 1024, "M": 1024**2, "G": 1024**3, "T": 1024**4}
        self.parameters = parameters
        self.run_bench = func
        self.space = []
        
        for param in parameters:
            if param.param_type in ["int", "float", "size"]:
                lower, upper = param.constraint
                if param.param_type == "size":
                    lower = lower * size_units[param.size_unit]
                    upper = upper * size_units[param.size_unit]
                if param.default_value < 100:
                    self.space.append(Real(lower, upper, name=param.name))
                else:
                    self.space.append(Real(math.log1p(lower), math.log1p(upper), name=param.name))
            elif param.param_type == "bool":
                self.space.append(Integer(0, 1, name=param.name))  # 0 代表 False，1 代表 True
            elif param.param_type == "str":
                self.space.append(Categorical(param.constraint, name=param.name))

    def encode_configs_as_vectors(self, x0: List[Dict[str, Union[str, int, float]]]):
        """
        Converts a list of configuration dictionaries into a list of vectors, 
        where each vector represents the values of the configuration parameters.

        Example:
            Input:
            x0 = [
                {"param1": 10, "param2": 20.5, "param3": "value1"},
                {"param1": 5, "param2": 15.0, "param3": "value2"}
            ]
            Output:
            [
                [10, 20.5, "value1"],
                [5, 15.0, "value2"]
            ]
        """
        processed_x0 = []
        for config in x0:
            vector = []
            for param_name, param_value in config.items():
                vector.append(param_value)
            processed_x0.append(vector)
        return processed_x0

    def validate_vectors(self, x0: List[List[Union[str, int, float]]]):
        """
        Validates configuration vectors against the defined parameter space.
        """
        for config in x0:
            for i, value in enumerate(config):
                if isinstance(self.space[i], Real):
                    if not (self.space[i].low <= value <= self.space[i].high):
                        raise ValueError(f"Initial configuration value {value} for parameter {self.space[i].name} is out of bounds.")
                elif isinstance(self.space[i], Integer):
                    if not (self.space[i].low <= value <= self.space[i].high):
                        raise ValueError(f"Initial configuration value {value} for parameter {self.space[i].name} is out of bounds.")
                elif isinstance(self.space[i], Categorical):
                    if value not in self.space[i].categories:
                        raise ValueError(f"Initial configuration value {value} for parameter {self.space[i].name} is not in the allowed categories.")

    def optimize(self, n_calls=30, x0=None, y0=None):
        # Step 1: 包装目标函数，适配 skopt 的输入格式
        @use_named_args(self.space)
        def run_bench_wrapper(**raw_params):
            processed_params = {}
            for param in self.parameters:
                value = raw_params.get(param.name)
                if value is None:
                    continue

                if param.param_type in ["int", "size"]:
                    # 对数缩放过的值，需要还原
                    processed_value = round(math.expm1(value)) if param.default_value > 100 else round(value)
                elif param.param_type == "float":
                    processed_value = math.expm1(value) if param.default_value > 100 else value
                elif param.param_type == "bool":
                    processed_value = bool(value)
                else:
                    processed_value = value

                processed_params[param.name] = processed_value

            result = self.run_bench(processed_params)
            print(f"Processed Parameters: {processed_params}, Result: {result}")
            return result

        # Step 2: 初始样本点预处理（编码向量 + 验证合法性）
        encoded_x0 = self.encode_configs_as_vectors(x0) if x0 else None
        if encoded_x0:
            self.validate_vectors(encoded_x0)

        # Step 3: 执行贝叶斯优化
        result = gp_minimize(
            func=run_bench_wrapper,
            dimensions=self.space,
            n_calls=n_calls,
            x0=encoded_x0,
            y0=y0,
            random_state=42
        )

        # Step 4: 解析最优参数并还原为原始格式
        best_params_vector = result.x
        best_params = {}

        for i, param in enumerate(self.parameters):
            value = best_params_vector[i]
            if param.param_type in ["int", "size"]:
                best_params[param.name] = round(math.expm1(value)) if param.default_value > 100 else round(value)
            elif param.param_type == "float":
                best_params[param.name] = math.expm1(value) if param.default_value > 100 else value
            elif param.param_type == "bool":
                best_params[param.name] = bool(value)
            else:
                best_params[param.name] = value

        return best_params, result.fun

def set_cephfs_config(config: Dict[str, Union[str, int, float]]) -> None:
    commands = []
    for key, value in config.items():
        commands.append(f"sudo ceph config set mds {key} {value}")
    full_command = " && ".join(commands)
    
    try:
        subprocess.run(full_command, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error setting Ceph configuration: {e}")

def run_ceph_benchmark(config: Dict[str, Union[str, int, float]]) -> float:
    set_cephfs_config(config)
    
    weights = {
        "Directory creation": 0.10,
        "Directory stat": 0.15,
        "Directory rename": 0.05,
        "Directory removal": 0.10,
        "File creation": 0.10,
        "File stat": 0.20,
        "File read": 0.10,
        "File removal": 0.10,
        "Tree creation": 0.07,
        "Tree removal": 0.03,
    }

    try:
        result = subprocess.run(
            ["sudo", "mdtest", "-d", "/mnt/cephfs", "-b", "6", "-I", "8", "-z", "2"],
            text=True,
            capture_output=True,
            check=True
        )

        op_pattern = re.compile(r"^\s*(.+?)\s+([\d.]+)\s+[\d.]+\s+([\d.]+)\s+[\d.]+")
        total_score = 0.0
        output_lines = result.stdout.split("\n")

        for line in output_lines:
            match = op_pattern.match(line)
            if match:
                operation = match.group(1).strip()
                mean_value = float(match.group(3))
                if operation in weights:
                    total_score += mean_value * weights[operation]

        return total_score

    except subprocess.CalledProcessError as e:
        print(f"Error running mdtest: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")

    return 0.0


params = CephParameterParser.parse_csv("mds-params.csv")

N_SAMPLE_CONFIGS = 1
sampled_configs = sample_configurations(params, N_SAMPLE_CONFIGS)
normalized_sampled_configs = [normalize_sampled_config(config, params) for config in sampled_configs]
sample_bhscores = [run_ceph_benchmark(config) for config in sampled_configs]

# extracted_feats = extract_important_features(
#                     [normalize_sampled_config(config, params) for config in sampled_configs],
#                     sample_bhscores
#                 )
# print(extracted_feats)

# print("Sampled Configurations:")
# for config in configs:
#     print(config)

optimizer = CephBayesianOptimizer(params, run_ceph_benchmark)
print(normalized_sampled_configs)
best_config, best_score = optimizer.optimize(x0=normalized_sampled_configs, y0=sample_bhscores)
print(f"Best Configuration: {best_config}")
