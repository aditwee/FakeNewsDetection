import os
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel


# ---- HARD DISABLE ALL ACCELERATION ----
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

# HARD disable MPS
torch.backends.mps.is_available = lambda: False

device = torch.device("cpu")
print("Using device:", device)

# ---- Model definition (MUST match training) ----
class DualInputDistilRoBERTa(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = self.encoder.config.hidden_size

        self.classifier = nn.Sequential(
            nn.Linear(hidden * 4, 512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, 1)
        )

    def forward(self, title_ids, title_mask, body_ids, body_mask):
        title_out = self.encoder(
            input_ids=title_ids,
            attention_mask=title_mask
        ).last_hidden_state[:, 0, :]

        body_out = self.encoder(
            input_ids=body_ids,
            attention_mask=body_mask
        ).last_hidden_state[:, 0, :]

        diff = torch.abs(title_out - body_out)
        prod = title_out * body_out

        fused = torch.cat([title_out, body_out, diff, prod], dim=1)
        return self.classifier(fused).squeeze(1)

# ---- Load checkpoint CORRECTLY ----
MODEL_PATH = "model.pt"

checkpoint = torch.load(MODEL_PATH, map_location="cpu")

model = DualInputDistilRoBERTa(checkpoint["model_name"])
model.load_state_dict(checkpoint["model_state_dict"])
model.to(device)
model.eval()

print("Model loaded successfully.")

# ---- Tokenizer (match training) ----
tokenizer = AutoTokenizer.from_pretrained(checkpoint["model_name"])

TITLE_MAX_LEN = 64
BODY_MAX_LEN = 384

# ---- Inference function (Flask-ready) ----
@torch.no_grad()
def predict_fake_news(title: str, body: str):
    title_enc = tokenizer(
        title,
        truncation=True,
        padding="max_length",
        max_length=TITLE_MAX_LEN,
        return_tensors="pt"
    )

    body_enc = tokenizer(
        body,
        truncation=True,
        padding="max_length",
        max_length=BODY_MAX_LEN,
        return_tensors="pt"
    )

    logits = model(
        title_enc["input_ids"].to(device),
        title_enc["attention_mask"].to(device),
        body_enc["input_ids"].to(device),
        body_enc["attention_mask"].to(device)
    )

    prob = torch.sigmoid(logits).item()
    label = "FAKE" if prob >= 0.5 else "REAL"

    return {
        "label": label,
        "probability": prob
    }


def main():
   
    title = "US President Donald Trump has signed an executive order aimed at blocking states from enforcing their own artificial intelligence (AI) regulations."
    body = """
    We want to have one central source of approval," Trump told reporters in the Oval Office on Thursday. It will give the Trump administration tools to push back on the most "onerous" state rules, said White House AI adviser David Sacks. The government will not oppose AI regulations around children's safety, he added. The move marks a win for technology giants who have called for US-wide AI legislation as it could have a major impact on America's goal of leading the fast-developing industry.AI company bosses have argued that state-level regulations could slow innovation and hinder the US in its race against China to dominate the industry, as firms pour billions of dollars into the technology. But the move to pre-empt state laws has drawn opposition. While the US currently has no national laws regulating AI, more than 1,000 separate AI bills have been introduced in states across the US, according to the White House. This year alone, 38 states including California, home to many of the world's biggest technology companies, have adopted some 100 AI regulations, the National Conference of State Legislatures says. Those rules range widely. One in California requires platforms to regularly remind users they are interacting with a chatbot, in a bid to protect children and teens from potential harms. The state also passed a bill requiring the largest AI developers to lay out plans to limit potential catastrophic risks stemming from their AI models. In North Dakota, a new law would prevent people from using AI-powered robots to stalk or harass others, while Arkansas bars AI content from infringing on intellectual property rights or existing copyright. Oregon brought in a law prohibiting a "non-human entity" including one powered by AI from using licenced medical titles, such as registered nurse. Critics of Trump's executive order have argued that state rules are necessary in the absence of meaningful guardrails at the federal level. "Stripping states from enacting their own AI safeguards undermines states' basic rights to establish sufficient guardrails to protect their residents," said Julie Scelfo, from advocacy group Mothers Against Media Addiction in a statement. California's Governor Gavin Newsom, a Democrat and vocal critic of the president, accused Trump of bowing to the interests of tech allies with the executive order. "Today, President Trump continued his ongoing grift in the White House, attempting to enrich himself and his associates, with a new executive order seeking to preempt state laws protecting Americans from unregulated AI technology," he said. AI firms OpenAI, Google, Meta, and Anthropic did not immediately respond to requests for comment on the order. But the move drew praise from tech lobby group NetChoice. "We look forward to working with the White House and Congress to set nationwide standards and a clear rulebook for innovators," said its director of policy Patrick Hedger. Michael Goodyear, an associate professor at New York Law School, said the AI industry was rightly alarmed about having to comply with a patchwork of rules, which might conflict. "It would be better to have one federal law than a bunch of conflicting state laws," he said. But, he added: "That assumes that we will have a good federal law in place.
    """
    print(predict_fake_news(title, body))

if __name__ == "__main__":
    main()
