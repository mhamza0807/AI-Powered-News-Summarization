from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch

DIR = "C:/Models/t5_model_final"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
my_model = T5ForConditionalGeneration.from_pretrained(DIR)
my_model = my_model.to(device)
my_tokenizer = T5Tokenizer.from_pretrained(DIR)

print("Model loaded successfully")

def generate_summary(article_text):
    input_text = f"summarize: {article_text}"

    inputs = my_tokenizer(input_text, return_tensors="pt", max_length=512,
                      truncation=True).to(device)

    summary_ids = my_model.generate(
        inputs["input_ids"],
        max_length=250,
        min_length=40,
        num_beams=4,
        early_stopping=True
    )

    summary = my_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

# Hardcoded article to test
article = """

GAZA CITY: Gaza’s civil defence agency said on Thursday Israeli bombardment killed at least 29 people since midnight in the war-ravaged territory, which has been under Israeli aid blockade for nearly two months.
Israeli Prime Minister Benjamin Netanyahu meanwhile said that while the military’s mission was to bring home all the prisoners from Gaza, its “supreme goal” was to achieve victory against Hamas.
Israel resumed its campaign in the Gaza Strip on March 18, after a two-month truce collapsed over disagreements between Israel and the Palestinian group Hamas.
Civil defence official Mohammed al-Mughayyir said Thursday’s toll included eight people killed in an air strike on the Abu Sahlul family home in Khan Yunis refugee camp in southern Gaza. Four people were killed in an air strike east of Shaaf in Gaza City’s Al-Tuffah neighbourhood, he said.
At least 17 more were killed in other attacks across the Palestinian territory, including one that hit a tent sheltering displaced people near the central city of Deir el-Balah, the agency said.
“We came here and found all these houses destroyed, and children, women and young people all bombed to pieces,” said Ahmed Abu Zarqa after a deadly strike in Khan Yunis. “This is no way to live. Enough, we’re tired, enough! “We don’t know what to do with our lives any more. We’d rather die than live this kind of life.”
AFP images showed residents digging through rubble in search of bodies, which were carried away on stretchers under blankets. At Nasser Hospital in Khan Yunis, rescuers rushed a screaming wounded child out of an ambulance.
The health ministry in Gaza said on Thursday that at least 2,326 people have been killed since Israel resumed strikes, bringing the overall death toll since the war broke out to 52,418.
The Hamas attack on Israel on Oct 7, 2023 resulted in the deaths of 1,218 people on the Israeli side, according to an AFP tally based on official figures.
Israel says its renewed military campaign aims to force Hamas to free the remaining prisoners.

‘Abomination’. The World Health Organisation decried on Thursday the horrifying situation unfolding in Gaza, with one top official voicing anger that the world was allowing the “abomination” to continue.
“We have to ask ourselves: How much blood is enough to satisfy whatever the political objectives are,” the UN health agency’s emergencies director Mike Ryan told reporters in Geneva.
“We are breaking the bodies and the minds of the children of Gaza. We are starving the children of Gaza, because if we don’t do something about it we are complicit in what is happening.”
Israel strictly controls all inflows of international aid vital for the 2.4 million Palestinians in the Gaza Strip.
It halted aid deliveries to Gaza on March 2, days before the collapse of a ceasefire that had significantly reduced hostilities after 15 months of war.
Since the start of the blockade, the UN has repeatedly warned of the humanitarian catastrophe on the ground, with famine again looming.
Supplies are dwindling and the UN’s World Food Programme (WFP) last Friday said it had sent out its “last remaining food stocks” to kitchens.
Ryan pointed to the more than 1,000 children in Gaza that are missing limbs, “thousands of children with spinal cord injuries, with severe head injuries from which they’ll never recover” and psychological conditions.


 """
# Generate and print the summary
summary = generate_summary(article)
print("\nARTICLE:")
print(article)
print("\nGENERATED SUMMARY:")
print(summary)