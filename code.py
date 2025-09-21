import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report, hamming_loss, jaccard_score
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModel, get_scheduler
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import lime
from lime.lime_text import LimeTextExplainer
import random
import re
import requests
from io import BytesIO
from zipfile import ZipFile
import torch.nn.init as init
from matplotlib.colors import LinearSegmentedColormap
import warnings
import shutil
warnings.filterwarnings("ignore")

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def verify_zip_structure(zip_path):
    print(f"Verifying ZIP structure: {zip_path}")
    required_files = [
        f"SemEval2025-Task11-main/task-dataset/semeval-2025-task11-dataset/track_a/{split}/{lang}.csv"
        for split in ["train", "dev", "test"]
        for lang in ["eng", "hin", "mar"]
    ]

    try:
        with ZipFile(zip_path, 'r') as zip_ref:
            zip_files = zip_ref.namelist()
            missing_files = [f for f in required_files if f not in zip_files]

            if missing_files:
                print("\nMissing files in ZIP:")
                for f in missing_files:
                    print(f"  {f}")
                return False
            else:
                print("\nAll required CSV files found in ZIP:")
                for f in required_files:
                    print(f"  {f}")
                return True
    except Exception as e:
        print(f"Error reading ZIP: {e}")
        return False

def verify_extracted_structure(base_dir="./data/track_a"):
    print(f"Verifying extracted dataset structure in {base_dir}...")
    required_files = [
        f"{base_dir}/{split}/{lang}/data.csv"
        for split in ["train", "dev", "test"]
        for lang in ["en", "hi", "mr"]
    ]

    missing_files = [f for f in required_files if not (os.path.exists(f) and os.path.getsize(f) > 0)]

    if missing_files:
        print("\nMissing or empty files:")
        for f in missing_files:
            print(f"  {f}")
        return False
    else:
        print("\nExtracted dataset structure is correct!")
        print("Found files:")
        for f in required_files:
            print(f"  {f}")
        return True

def extract_dataset_from_zip(zip_path):
    print(f"Extracting dataset from {zip_path}...")
    try:
        if not verify_zip_structure(zip_path):
            print("ZIP file is missing required files.")
            return False

        with ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall("./data/tmp")

        src_base = "./data/tmp/SemEval2025-Task11-main/task-dataset/semeval-2025-task11-dataset/track_a"
        dst_base = "./data/track_a"

        lang_map = {"eng": "en", "hin": "hi", "mar": "mr"}
        splits = ["train", "dev", "test"]

        for split in splits:
            for src_lang, dst_lang in lang_map.items():
                src_file = os.path.join(src_base, split, f"{src_lang}.csv")
                dst_dir = os.path.join(dst_base, split, dst_lang)
                dst_file = os.path.join(dst_dir, "data.csv")

                if os.path.exists(src_file):
                    os.makedirs(dst_dir, exist_ok=True)
                    shutil.copy2(src_file, dst_file)
                    print(f"Copied {src_file} to {dst_file}")
                else:
                    print(f"Warning: {src_file} not found in ZIP")

        shutil.rmtree("./data/tmp")
        print("Dataset extracted successfully!")

        if not verify_extracted_structure():
            print("Extracted structure is incomplete.")
            return False
        return True
    except Exception as e:
        print(f"Failed to extract dataset: {e}")
        return False

def ensure_dataset(zip_path, language='en'):
    required_files = [
        f"./data/track_a/{split}/{language}/data.csv"
        for split in ["train", "dev", "test"]
    ]
    all_files_exist = all(os.path.exists(f) and os.path.getsize(f) > 0 for f in required_files)

    if not all_files_exist:
        print(f"Some dataset files for language {language} are missing or empty.")
        if not os.path.exists(zip_path):
            print(f"ZIP file not found at {zip_path}. Please download SemEval2025-Task11-main.zip from "
                  "https://github.com/emotion-analysis-project/SemEval2025-Task11 and specify the correct path.")
            return False
        success = extract_dataset_from_zip(zip_path)
        if not success:
            print("Dataset extraction failed.")
            return False
    else:
        if not verify_extracted_structure():
            print("Existing dataset structure is incomplete. Attempting to re-extract...")
            success = extract_dataset_from_zip(zip_path)
            if not success:
                print("Dataset extraction failed.")
                return False
    return True

def translate_text(text, source_lang, target_lang):
    if source_lang == target_lang or target_lang in ['hi', 'mr']:
        return text
    return f"[{target_lang}] {text}"

def load_dataset(language='en', augment_with_translations=False, languages=None, all_emotion_cols=None):
    splits = ["train", "dev"]
    all_dfs = []

    for split in splits:
        csv_file = f"./data/track_a/{split}/{language}/data.csv"
        if not os.path.exists(csv_file):
            print(f"File not found: {csv_file}")
            return None

        df = pd.read_csv(csv_file)
        all_dfs.append(df)

    combined_df = pd.concat(all_dfs, ignore_index=True)

    emotion_cols = [col for col in combined_df.columns if col not in ['id', 'text']]

    for col in emotion_cols:
        if combined_df[col].isna().any():
            print(f"Warning: NaN values found in {col} for language {language}. Filling with 0.")
            combined_df[col] = combined_df[col].fillna(0)
        if not pd.api.types.is_numeric_dtype(combined_df[col]):
            print(f"Warning: Non-numeric values in {col} for language {language}. Converting to numeric.")
            combined_df[col] = pd.to_numeric(combined_df[col], errors='coerce').fillna(0)
        combined_df[col] = combined_df[col].clip(0, 1).astype(int)

    if all_emotion_cols:
        for col in all_emotion_cols:
            if col not in combined_df.columns:
                print(f"Adding missing emotion {col} for language {language} with zeros.")
                combined_df[col] = 0
        emotion_cols = all_emotion_cols
        combined_df = combined_df[['id', 'text'] + emotion_cols]

    if augment_with_translations and language != 'en' and languages:
        en_df = []
        for split in splits:
            en_csv = f"./data/track_a/{split}/en/data.csv"
            if os.path.exists(en_csv):
                df = pd.read_csv(en_csv)
                for col in emotion_cols:
                    if col in df.columns:
                        if df[col].isna().any():
                            print(f"Warning: NaN values in {col} for English augmentation. Filling with 0.")
                            df[col] = df[col].fillna(0)
                        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).clip(0, 1).astype(int)
                    else:
                        df[col] = 0
                en_df.append(df)
        if en_df:
            en_df = pd.concat(en_df, ignore_index=True)
            en_df['text'] = en_df['text'].apply(lambda x: translate_text(x, 'en', language))
            for col in emotion_cols:
                if col not in en_df.columns:
                    en_df[col] = 0
            en_df = en_df[['id', 'text'] + emotion_cols]
            combined_df = pd.concat([combined_df, en_df], ignore_index=True)

    print(f"\nDataset for language '{language}':")
    print(f"Shape: {combined_df.shape}")
    print("Columns:", combined_df.columns.tolist())
    print("\nClass distribution:")
    for emotion in emotion_cols:
        count = combined_df[emotion].sum()
        print(f"{emotion}: {count} ({count/len(combined_df):.2%})")

    return combined_df, emotion_cols

class EmotionDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten().to(device),
            'attention_mask': encoding['attention_mask'].flatten().to(device),
            'labels': torch.FloatTensor(label).to(device)
        }

class EmotionClassifier(nn.Module):
    def __init__(self, model_name, num_labels, dropout_prob=0.1):
        super(EmotionClassifier, self).__init__()
        self.transformer = AutoModel.from_pretrained(model_name).to(device)
        self.dropout = nn.Dropout(dropout_prob)

        hidden_size = self.transformer.config.hidden_size

        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size // 2,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout_prob if 2 > 1 else 0
        ).to(device)

        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False),
            nn.Softmax(dim=1)
        ).to(device)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_size // 2, num_labels)
        ).to(device)

        self._init_weights(self.classifier)
        self._init_weights(self.attention)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            init.xavier_uniform_(module.weight)
            if module.bias is not None:
                init.zeros_(module.bias)

    def forward(self, input_ids, attention_mask, return_attentions=False):
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        hidden_states = outputs.last_hidden_state

        lstm_out, _ = self.lstm(hidden_states)

        attention_weights = self.attention(lstm_out)
        context_vector = torch.sum(attention_weights * lstm_out, dim=1)

        pooled_output = self.dropout(context_vector)
        logits = self.classifier(pooled_output)

        if return_attentions:
            return logits, attention_weights, None

        return logits

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.eps = 1e-8

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt + self.eps) ** self.gamma * BCE_loss
        F_loss = torch.clamp(F_loss, min=-1e10, max=1e10)
        if torch.isnan(F_loss).any():
            print("Warning: NaN detected in FocalLoss")
        if self.reduction == 'mean':
            return F_loss.mean()
        elif self.reduction == 'sum':
            return F_loss.sum()
        else:
            return F_loss

class WeightedBCELoss(nn.Module):
    def __init__(self, samples_per_class):
        super(WeightedBCELoss, self).__init__()
        weights = torch.tensor([1.0 / (s + 1e-10) for s in samples_per_class], dtype=torch.float32).to(device)
        self.weights = weights / (weights.sum() + 1e-10) * len(samples_per_class)
        self.eps = 1e-8

    def forward(self, inputs, targets):
        bce = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        weighted_bce = bce * self.weights[None, :]
        weighted_bce = torch.clamp(weighted_bce, min=-1e10, max=1e10)
        if torch.isnan(weighted_bce).any():
            print("Warning: NaN detected in WeightedBCELoss")
        return weighted_bce.mean()

class CometInspiredLoss(nn.Module):
    def __init__(self, samples_per_class_per_lang, languages, alpha=2, gamma=2):
        super(CometInspiredLoss, self).__init__()
        self.focal_loss = FocalLoss(alpha=alpha, gamma=gamma)
        self.languages = languages
        self.eps = 1e-8
        weights = []
        for lang_samples in samples_per_class_per_lang:
            lang_weights = torch.tensor([1.0 / (s + self.eps) for s in lang_samples], dtype=torch.float32)
            weights.append(lang_weights / (lang_weights.sum() + self.eps) * len(lang_samples))
        self.weights = torch.stack(weights).mean(dim=0).to(device)
        self.weights = torch.clamp(self.weights, min=self.eps, max=1e10)

    def forward(self, inputs, targets):
        focal_loss = self.focal_loss(inputs, targets)
        weighted_loss = focal_loss * self.weights[None, :]
        if torch.isnan(weighted_loss).any():
            print("Warning: NaN detected in CometInspiredLoss")
        return weighted_loss.mean()

class AsymmetricFocalLoss(nn.Module):
    def __init__(self, gamma_pos=1, gamma_neg=3, alpha=0.25, reduction='mean'):
        super(AsymmetricFocalLoss, self).__init__()
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.alpha = alpha
        self.reduction = reduction
        self.eps = 1e-8

    def forward(self, inputs, targets):
        probs = torch.sigmoid(inputs)
        probs = torch.clamp(probs, min=self.eps, max=1 - self.eps)
        loss_pos = -self.alpha * (1 - probs) ** self.gamma_pos * targets * torch.log(probs)
        loss_neg = -(1 - self.alpha) * probs ** self.gamma_neg * (1 - targets) * torch.log(1 - probs)
        loss = loss_pos + loss_neg
        loss = torch.clamp(loss, min=-1e10, max=1e10)
        if torch.isnan(loss).any():
            print("Warning: NaN detected in AsymmetricFocalLoss")
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

def train_epoch(model, data_loader, optimizer, scheduler, loss_fn):
    model.train()
    total_loss = 0
    progress_bar = tqdm(data_loader, desc="Training")
    for batch in progress_bar:
        optimizer.zero_grad()

        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = loss_fn(outputs, labels)

        if torch.isnan(loss):
            print("Warning: NaN loss detected, skipping batch")
            continue

        total_loss += loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        progress_bar.set_postfix({'loss': loss.item()})

    return total_loss / len(data_loader)

def evaluate(model, data_loader, loss_fn, emotion_labels):
    model.eval()
    val_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = loss_fn(outputs, labels)
            val_loss += loss.item()

            probs = torch.sigmoid(outputs)
            preds = (probs >= 0.5).float().cpu().numpy()
            labels_np = labels.cpu().numpy()

            all_preds.extend(preds)
            all_labels.extend(labels_np)

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    if np.isnan(all_labels).any():
        print("Error: NaN values in true labels during evaluation")
        raise ValueError("Input y_true contains NaN")

    f1_macro = f1_score(all_labels, all_preds, average='macro')
    precision_macro = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall_macro = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    hamming = hamming_loss(all_labels, all_preds)
    jaccard = jaccard_score(all_labels, all_preds, average='samples')

    report = classification_report(all_labels, all_preds, target_names=emotion_labels, zero_division=0)

    metrics = {
        'val_loss': val_loss / len(data_loader),
        'f1_macro': f1_macro,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'hamming_loss': hamming,
        'jaccard_score': jaccard
    }

    return metrics, report, all_preds, all_labels

def explain_with_lime(model, tokenizer, text, emotion_labels, num_features=10):
    def predict_proba(texts):
        model.eval()
        results = []

        for text in texts:
            encoding = tokenizer(
                text,
                add_special_tokens=True,
                max_length=128,
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
                return_tensors='pt'
            )

            input_ids = encoding['input_ids'].to(device)
            attention_mask = encoding['attention_mask'].to(device)

            with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                probs = torch.sigmoid(outputs).cpu().numpy()[0]

                result = np.stack([(1 - probs), probs], axis=1)
                results.append(result.flatten())

        return np.array(results)

    explainer = LimeTextExplainer(class_names=emotion_labels)

    exp = explainer.explain_instance(
        text,
        predict_proba,
        num_features=num_features,
        num_samples=500,
        labels=range(len(emotion_labels))
    )

    explanations = {}
    for i, emotion in enumerate(emotion_labels):
        explanations[emotion] = exp.as_list(label=i)

    return explanations, exp

def visualize_attention(model, tokenizer, text, emotion_labels):
    tokens = tokenizer.tokenize(tokenizer.decode(tokenizer.encode(text)))

    encoding = tokenizer(
        text,
        add_special_tokens=True,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )

    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    model.eval()
    with torch.no_grad():
        logits, attn_weights, _ = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_attentions=True
        )

        probs = torch.sigmoid(logits).cpu().numpy()[0]

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = [(1, 1, 1), (0, 0, 1)]
    cmap = LinearSegmentedColormap.from_list("white_to_blue", colors, N=100)

    attn = attn_weights.cpu().numpy()[0][:len(tokens)]

    df = pd.DataFrame({'Token': tokens, 'Weight': attn.flatten()})
    df = df.set_index('Token')

    sns.heatmap(df.T, cmap=cmap, ax=ax, cbar=True)
    ax.set_title(f"Attention Weights (Probabilities: {', '.join([f'{e}: {p:.2f}' for e, p in zip(emotion_labels, probs)])})")
    ax.set_xlabel('Tokens')
    ax.set_yticklabels([])

    plt.tight_layout()
    return fig

def train_multilingual(languages, model_name="bert-base-multilingual-uncased", num_epochs=5, batch_size=16, learning_rate=2e-5, loss_type='focal'):
    print(f"\n{'='*50}")
    print(f"Training multilingual model for languages: {', '.join(languages)}")
    print(f"Using model: {model_name}")
    print(f"Loss function: {loss_type}")
    print(f"{'='*50}")

    ZIP_PATH = "/content/SemEval2025-Task11-main.zip"
    combined_data = []
    all_emotion_cols = set()
    samples_per_class_per_lang = []

    for language in languages:
        if not ensure_dataset(ZIP_PATH, language):
            return None
        data, emotion_cols = load_dataset(language, augment_with_translations=False)
        if data is None:
            return None
        all_emotion_cols.update(emotion_cols)

    all_emotion_cols = sorted(list(all_emotion_cols))

    for language in languages:
        data, _ = load_dataset(
            language,
            augment_with_translations=True,
            languages=languages,
            all_emotion_cols=all_emotion_cols
        )
        if data is None:
            return None
        combined_data.append(data)
        samples_per_class = [data[emotion].sum() for emotion in all_emotion_cols]
        samples_per_class_per_lang.append(samples_per_class)

    data = pd.concat(combined_data, ignore_index=True)

    train_df, test_df = train_test_split(data, test_size=0.2, random_state=42)
    train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=42)

    print(f"Training set: {len(train_df)} samples")
    print(f"Validation set: {len(val_df)} samples")
    print(f"Test set: {len(test_df)} samples")

    samples_per_class = [train_df[emotion].sum() for emotion in all_emotion_cols]

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    train_dataset = EmotionDataset(
        train_df['text'].values,
        train_df[all_emotion_cols].values,
        tokenizer
    )

    val_dataset = EmotionDataset(
        val_df['text'].values,
        val_df[all_emotion_cols].values,
        tokenizer
    )

    test_dataset = EmotionDataset(
        test_df['text'].values,
        test_df[all_emotion_cols].values,
        tokenizer
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    model = EmotionClassifier(model_name, len(all_emotion_cols)).to(device)

    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)

    total_steps = len(train_loader) * num_epochs

    scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )

    if loss_type == 'focal':
        loss_fn = FocalLoss(alpha=2, gamma=2)
    elif loss_type == 'weighted_bce':
        loss_fn = WeightedBCELoss(samples_per_class)
    elif loss_type == 'comet':
        loss_fn = CometInspiredLoss(samples_per_class_per_lang, languages)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")

    best_f1 = 0
    best_model_state = None
    training_history = {'train_loss': [], 'val_loss': [], 'f1_macro': [], 'precision_macro': [], 'recall_macro': [], 'hamming_loss': [], 'jaccard_score': []}

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, loss_fn)
        training_history['train_loss'].append(train_loss)

        val_metrics, val_report, _, _ = evaluate(model, val_loader, loss_fn, all_emotion_cols)

        training_history['val_loss'].append(val_metrics['val_loss'])
        training_history['f1_macro'].append(val_metrics['f1_macro'])
        training_history['precision_macro'].append(val_metrics['precision_macro'])
        training_history['recall_macro'].append(val_metrics['recall_macro'])
        training_history['hamming_loss'].append(val_metrics['hamming_loss'])
        training_history['jaccard_score'].append(val_metrics['jaccard_score'])

        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_metrics['val_loss']:.4f}")
        print(f"F1 Macro: {val_metrics['f1_macro']:.4f}")
        print(f"Precision Macro: {val_metrics['precision_macro']:.4f}")
        print(f"Recall Macro: {val_metrics['recall_macro']:.4f}")
        print(f"Hamming Loss: {val_metrics['hamming_loss']:.4f}")
        print(f"Jaccard Score: {val_metrics['jaccard_score']:.4f}")

        if val_metrics['f1_macro'] > best_f1:
            best_f1 = val_metrics['f1_macro']
            best_model_state = model.state_dict().copy()
            print(f"New best model saved with F1 Macro: {best_f1:.4f}")

    if best_model_state:
        model.load_state_dict(best_model_state)

    test_metrics, test_report, test_predictions, test_labels = evaluate(model, test_loader, loss_fn, all_emotion_cols)

    print("\nTest Results:")
    print(f"F1 Macro: {test_metrics['f1_macro']:.4f}")
    print(f"Precision Macro: {test_metrics['precision_macro']:.4f}")
    print(f"Recall Macro: {test_metrics['recall_macro']:.4f}")
    print(f"Hamming Loss: {test_metrics['hamming_loss']:.4f}")
    print(f"Jaccard Score: {test_metrics['jaccard_score']:.4f}")
    print("\nDetailed Classification Report:")
    print(test_report)

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(training_history['train_loss'], label='Train Loss')
    plt.plot(training_history['val_loss'], label='Val Loss')
    plt.title('Loss History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(training_history['f1_macro'], label='F1 Macro')
    plt.plot(training_history['precision_macro'], label='Precision Macro')
    plt.plot(training_history['recall_macro'], label='Recall Macro')
    plt.title('Metrics History')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(15, 10))
    n_emotions = len(all_emotion_cols)
    rows = (n_emotions + 1) // 2
    cols = 2 if n_emotions > 1 else 1

    for i, emotion in enumerate(all_emotion_cols):
        plt.subplot(rows, cols, i+1)
        true_pos = ((test_predictions[:, i] == 1) & (test_labels[:, i] == 1)).sum()
        false_pos = ((test_predictions[:, i] == 1) & (test_labels[:, i] == 0)).sum()
        true_neg = ((test_predictions[:, i] == 0) & (test_labels[:, i] == 0)).sum()
        false_neg = ((test_predictions[:, i] == 0) & (test_labels[:, i] == 1)).sum()

        cm = np.array([[true_neg, false_pos], [false_neg, true_pos]])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Negative', 'Positive'],
                   yticklabels=['Negative', 'Positive'])
        plt.title(f'Confusion Matrix - {emotion}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')

    plt.tight_layout()
    plt.show()

    sample_indices = []
    for i in range(n_emotions):
        indices = np.where(test_labels[:, i] == 1)[0]
        if len(indices) > 0:
            sample_indices.append(random.choice(indices))

    sample_texts = test_df.iloc[sample_indices]['text'].values

    for i, text in enumerate(sample_texts[:5]):
        print(f"\nAnalyzing sample text for interpretability (LIME):")
        print(f"Text: {text}")
        emotion_idx = i % n_emotions
        emotion = all_emotion_cols[emotion_idx]
        print(f"Target emotion: {emotion}")

        explanations, exp = explain_with_lime(model, tokenizer, text, all_emotion_cols)
        for emotion, explanation in explanations.items():
            print(f"\nExplanation for {emotion}:")
            for word, score in explanation:
                print(f"  {word}: {score:.4f}")

        plt.figure(figsize=(10, 4))
        exp.as_pyplot_figure(label=emotion_idx)
        plt.title(f"LIME explanation for {emotion}")
        plt.show()

        print("\nVisualizing attention weights:")
        fig = visualize_attention(model, tokenizer, text, all_emotion_cols)
        plt.show()

    return model, tokenizer, test_df, all_emotion_cols

def train_and_evaluate(language, model_name, num_epochs=5, batch_size=16, learning_rate=2e-5, loss_type='focal'):
    print(f"\n{'='*50}")
    print(f"Training model for language: {language}")
    print(f"Using model: {model_name}")
    print(f"Loss function: {loss_type}")
    print(f"{'='*50}")

    ZIP_PATH = "/content/SemEval2025-Task11-main.zip"
    if not ensure_dataset(ZIP_PATH, language):
        return None

    data_result = load_dataset(language)
    if data_result is None:
        return None

    data, emotion_labels = data_result

    train_df, test_df = train_test_split(data, test_size=0.2, random_state=42)
    train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=42)

    print(f"Training set: {len(train_df)} samples")
    print(f"Validation set: {len(val_df)} samples")
    print(f"Test set: {len(test_df)} samples")

    samples_per_class = [train_df[emotion].sum() for emotion in emotion_labels]

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    train_dataset = EmotionDataset(
        train_df['text'].values,
        train_df[emotion_labels].values,
        tokenizer
    )

    val_dataset = EmotionDataset(
        val_df['text'].values,
        val_df[emotion_labels].values,
        tokenizer
    )

    test_dataset = EmotionDataset(
        test_df['text'].values,
        test_df[emotion_labels].values,
        tokenizer
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    model = EmotionClassifier(model_name, len(emotion_labels))
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)

    total_steps = len(train_loader) * num_epochs

    scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )

    if loss_type == 'focal':
        loss_fn = FocalLoss(alpha=2, gamma=2)
    elif loss_type == 'weighted_bce':
        loss_fn = WeightedBCELoss(samples_per_class)
    elif loss_type == 'asymmetric_focal':
        loss_fn = AsymmetricFocalLoss(gamma_pos=1, gamma_neg=3, alpha=0.25)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")

    best_f1 = 0
    best_model_state = None
    training_history = {'train_loss': [], 'val_loss': [], 'f1_macro': [], 'precision_macro': [], 'recall_macro': [], 'hamming_loss': [], 'jaccard_score': []}

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, loss_fn)
        training_history['train_loss'].append(train_loss)

        val_metrics, val_report, _, _ = evaluate(model, val_loader, loss_fn, emotion_labels)

        training_history['val_loss'].append(val_metrics['val_loss'])
        training_history['f1_macro'].append(val_metrics['f1_macro'])
        training_history['precision_macro'].append(val_metrics['precision_macro'])
        training_history['recall_macro'].append(val_metrics['recall_macro'])
        training_history['hamming_loss'].append(val_metrics['hamming_loss'])
        training_history['jaccard_score'].append(val_metrics['jaccard_score'])

        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_metrics['val_loss']:.4f}")
        print(f"F1 Macro: {val_metrics['f1_macro']:.4f}")
        print(f"Precision Macro: {val_metrics['precision_macro']:.4f}")
        print(f"Recall Macro: {val_metrics['recall_macro']:.4f}")
        print(f"Hamming Loss: {val_metrics['hamming_loss']:.4f}")
        print(f"Jaccard Score: {val_metrics['jaccard_score']:.4f}")

        if val_metrics['f1_macro'] > best_f1:
            best_f1 = val_metrics['f1_macro']
            best_model_state = model.state_dict().copy()
            print(f"New best model saved with F1 Macro: {best_f1:.4f}")

    if best_model_state:
        model.load_state_dict(best_model_state)

    test_metrics, test_report, test_predictions, test_labels = evaluate(model, test_loader, loss_fn, emotion_labels)

    print("\nTest Results:")
    print(f"F1 Macro: {test_metrics['f1_macro']:.4f}")
    print(f"Precision Macro: {test_metrics['precision_macro']:.4f}")
    print(f"Recall Macro: {test_metrics['recall_macro']:.4f}")
    print(f"Hamming Loss: {test_metrics['hamming_loss']:.4f}")
    print(f"Jaccard Score: {test_metrics['jaccard_score']:.4f}")
    print("\nDetailed Classification Report:")
    print(test_report)

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(training_history['train_loss'], label='Train Loss')
    plt.plot(training_history['val_loss'], label='Val Loss')
    plt.title('Loss History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(training_history['f1_macro'], label='F1 Macro')
    plt.plot(training_history['precision_macro'], label='Precision Macro')
    plt.plot(training_history['recall_macro'], label='Recall Macro')
    plt.title('Metrics History')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(15, 10))
    n_emotions = len(emotion_labels)
    rows = (n_emotions + 1) // 2
    cols = 2 if n_emotions > 1 else 1

    for i, emotion in enumerate(emotion_labels):
        plt.subplot(rows, cols, i+1)
        true_pos = ((test_predictions[:, i] == 1) & (test_labels[:, i] == 1)).sum()
        false_pos = ((test_predictions[:, i] == 1) & (test_labels[:, i] == 0)).sum()
        true_neg = ((test_predictions[:, i] == 0) & (test_labels[:, i] == 0)).sum()
        false_neg = ((test_predictions[:, i] == 0) & (test_labels[:, i] == 1)).sum()

        cm = np.array([[true_neg, false_pos], [false_neg, true_pos]])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Negative', 'Positive'],
                   yticklabels=['Negative', 'Positive'])
        plt.title(f'Confusion Matrix - {emotion}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')

    plt.tight_layout()
    plt.show()

    sample_indices = []
    for i in range(n_emotions):
        indices = np.where(test_labels[:, i] == 1)[0]
        if len(indices) > 0:
            sample_indices.append(random.choice(indices))

    sample_texts = test_df.iloc[sample_indices]['text'].values

    for i, text in enumerate(sample_texts[:5]):
        print(f"\nAnalyzing sample text for interpretability (LIME):")
        print(f"Text: {text}")
        emotion_idx = i % n_emotions
        emotion = emotion_labels[emotion_idx]
        print(f"Target emotion: {emotion}")

        explanations, exp = explain_with_lime(model, tokenizer, text, emotion_labels)
        for emotion, explanation in explanations.items():
            print(f"\nExplanation for {emotion}:")
            for word, score in explanation:
                print(f"  {word}: {score:.4f}")

        plt.figure(figsize=(10, 4))
        exp.as_pyplot_figure(label=emotion_idx)
        plt.title(f"LIME explanation for {emotion}")
        plt.show()

        print("\nVisualizing attention weights:")
        fig = visualize_attention(model, tokenizer, text, emotion_labels)
        plt.show()

    return model, tokenizer, test_df, emotion_labels

if __name__ == "__main__":
    print("Ensure all required packages are installed: torch, transformers, pandas, numpy, sklearn, matplotlib, seaborn, tqdm, lime")
    print("Run in Google Colab with GPU enabled for best performance.")

    languages = ['en', 'hi', 'mr']
    loss_functions_en = ['focal', 'weighted_bce', 'asymmetric_focal']
    loss_functions_multi = ['focal', 'weighted_bce', 'comet']

    models = {}

    for loss_type in loss_functions_en:
        model_results = train_and_evaluate(
            language='en',
            model_name="distilbert-base-uncased",
            num_epochs=5,
            batch_size=16,
            learning_rate=2e-5,
            loss_type=loss_type
        )
        if model_results:
            models[f'en_{loss_type}'] = model_results

    for loss_type in loss_functions_multi:
        model_results = train_multilingual(
            languages=['hi', 'mr'],
            model_name="bert-base-multilingual-uncased",
            num_epochs=5,
            batch_size=16,
            learning_rate=2e-5,
            loss_type=loss_type
        )
        if model_results:
            models[f'multilingual_{loss_type}'] = model_results

    for key, (model, tokenizer, test_df, emotion_labels) in models.items():
        print(f"\n{'='*50}")
        print(f"Testing model for: {key}")
        print(f"{'='*50}")

        if key.startswith('multilingual'):
            native_df = test_df[~test_df['text'].str.startswith('[hi]') & ~test_df['text'].str.startswith('[mr]')]
            if len(native_df) >= 5:
                sample_texts = native_df['text'].sample(5, random_state=42).values
            else:
                print(f"Warning: Not enough native texts for {key}. Using mixed sampling.")
                sample_texts = test_df['text'].sample(5, random_state=42).values
        else:
            sample_texts = test_df['text'].sample(5, random_state=42).values

        print("\nTesting with example texts:")
        for text in sample_texts:
            encoding = tokenizer(
                text,
                add_special_tokens=True,
                max_length=128,
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
                return_tensors='pt'
            )

            input_ids = encoding['input_ids'].to(device)
            attention_mask = encoding['attention_mask'].to(device)

            model.eval()
            with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                probs = torch.sigmoid(outputs).cpu().numpy()[0]
                preds = (probs >= 0.5).astype(int)

            print(f"\nText: {text}")
            print(f"Predicted emotions: {preds}")
            print("Probabilities:")
            for emotion, prob in zip(emotion_labels, probs):
                print(f"  {emotion}: {prob:.4f}")

            detected = [emotion_labels[i] for i, p in enumerate(preds) if p == 1]
            if detected:
                print(f"Detected emotions: {', '.join(detected)}")
            else:
                print("No emotions detected.")

    print("\nProject completed successfully!")
