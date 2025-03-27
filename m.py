import chromadb
from sentence_transformers import SentenceTransformer

# Initialize ChromaDB client
chroma_client = chromadb.PersistentClient(path="./chroma_db")  # Persist data

# Create a collection (or get an existing one)
collection = chroma_client.get_or_create_collection(name="rag_docs")

# Load a sentence transformer model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Sample documents
documents = [
    "Tag: greeting | Patterns: Hi, Hey, Hello | Responses: Hi! I'm QuerEase, what can I help you with?, Hi! How can I assist you today?",
    "Tag: goodbye | Patterns: Bye, See you later, Goodbye | Responses: See you later, thanks for visiting, Have a nice day, Bye! Come back again soon.",
    "Tag: thanks | Patterns: Thanks, Thank you, That's helpful, Thank's a lot! | Responses: Happy to help!, Any time!, My pleasure",
    "Tag: Customer Care | Patterns: Swiggy Customer Care number please, What is Swiggy Customer Care Number?, Can I know what Swiggy Customer Care number is? | Responses: We value our customer's time and hence moved away from a single customer care number to a comprehensive chat-based support system, We currently have moved out of a single customer care to comprehensive chat-based support.",
    "Tag: Surge Fee | Patterns: When is surge fee applicable, I see surge fees on app, What is surge fee | Responses: Surge fee is generally enabled temporarily due to higher than expected demand to help us fairly compensate the delivery executive., It is a fee that fairly goes to delivery executives if there is higher demand than expected",
    "Tag: Login issue | Patterns: An OTP is not being sent, I did not receive my OTP on SMS, I am not receiving an OTP | Responses: If you're not receiving the OTP, it's usually due to a network issue., Please check your mobile network settings and try generating a new OTP.",
    "Tag: Wrong CVV | Patterns: Will my transaction proceed if I enter a wrong OTP, I entered the wrong CVV, why did my transaction still go through?, I entered wrong CVV | Responses: The logic of validations of CVV resides with either payment gateways or banks., It is absolutely the choice of banks to have CVV as a mandatory input field or not.",
    "Tag: Order modify | Patterns: Does Swiggy allow me to edit my order, Can I edit my order?, Can I modify my order? | Responses: In order to edit your order, please click on 'Help' and then 'I want to modify items in my order'., We will connect you to a support agent who will assist you with the same.",
    "Tag: Quality/Quantity | Patterns: Who is responsible for Quality, Will Swiggy be accountable for quality, Will Swiggy be accountable for quantity | Responses: Quantity and quality of the food is the restaurant's responsibility., In case of issues with the quality or quantity, kindly submit your feedback and we will pass it on to the restaurant.",
    "Tag: Minimum Order | Patterns: Minimum order Value?, Is there a minimum order value?, Is there any minimum order limit? | Responses: We have no minimum order value, You can order for any amount.",
    "Tag: Delivery Fee | Patterns: What are delivery charges, Do you charge for delivery?, What is the delivery Fee? | Responses: Delivery fee varies from city to city and is applicable if order value is below a certain amount., Certain restaurants might have fixed delivery fees and for some charges apply as per the order quantity.",
    "Tag: Restaurant Location | Patterns: Can I order from any restaurant from any hotel, Can I order from any location?, Can I order from any Restaurant? | Responses: We will deliver from any restaurant listed on the search results for your location., We recommend enabling your GPS location finder and letting the app auto-detect your location to see all the available restaurants.",
    "Tag: Bulk Orders | Patterns: Are Bulk orders accepted, Do you support bulk orders?, Do you deliver Bulk orders? | Responses: In order to provide all customers with a great selection and to ensure on-time delivery of your meal, we reserve the right to limit the quantities depending on supply.",
    "Tag: Pre Order | Patterns: Is advance ordering possible, Can I order in advance?, Can I pre-order in advance? | Responses: Yes, you can order up to 2 days in advance on our platform. Click on the 'NOW' button on the top left corner of the app to select your desired delivery time slot and place an order. This feature is currently available only on Android phones and in select cities.",
    "Tag: Modify address/number | Patterns: Can I change my address after placing an order, Is it possible to change my address, Can I change my address, Can I change my number | Responses: Any major change in delivery address is not possible after you have placed an order with us. However, slight modifications like changing the flat number, street name, landmark etc. are allowed. If you have received delivery executive details, you can directly call him, else you could contact our customer service team.",
    "Tag: Profile details | Patterns: I cannot see my profile details, Unable to view the details in my profile, Unable to see the details in my profile | Responses: Please check if your app is due for an update. If not, please share the details via support@swiggy.in",
    "Tag: Account Deactivation | Patterns: How can I Deactivate my account, Deactivate my account, How to deactivate an account, Can I deactivate my account? | Responses: Please write to us at support@swiggy.in in the event that you want to deactivate your account.",
    "Tag: Swiggy One Limit | Patterns: Is there a limit on the number of devices I can use Swiggy One on?, How many devices can I use Swiggy One on?, Can I use Swiggy One on more than 2 devices? | Responses: Yes. Swiggy One membership can be used only on 2 devices at a time from 8th Feb onwards.",
    "Tag: Onboarding Fee | Patterns: Will you let me know what onboarding fee is, Can I know what onboarding fee is?, What is this one-time Onboarding fee? | Responses: This is a one-time fee charged towards the system & admin costs incurred during the onboarding process. It is deducted from the weekly payouts after you start receiving orders from Swiggy.",
    "Tag: Commission charges | Patterns: What are your commission charges, How much commission do you charge, How much commission will I be charged by Swiggy | Responses: The commission charges vary for different cities. You will be able to see the commission applicable for you once the preliminary onboarding details have been filled.",
    "Tag: FSSAI licence | Patterns: Can I onboard on Swiggy without an FSSAI licence, I don't have an FSSAI Licence can I still onboard?, I don't have an FSSAI licence for my restaurant. Can it still be onboarded? | Responses: FSSAI licence is a mandatory requirement according to the government's policies. However, if you are yet to receive the licence at the time of onboarding, you can proceed with the acknowledgement number which you will have received from FSSAI for your registration.",
    "Tag: IRCTC Delivery | Patterns: What happens if my train gets cancelled, What happens to my order if my train gets delayed, What will happen if the train is delayed? | Responses: We'll be checking the status of the train on a timely basis and in case the train is delayed, we'll reschedule the order based on the train's arrival time."
]





# Generate embeddings and store in ChromaDB
for i, doc in enumerate(documents):
    embedding = embedding_model.encode(doc).tolist()  # Convert to list for ChromaDB
    collection.add(ids=[f"doc_{i}"], embeddings=[embedding], metadatas=[{"text": doc}])

print("Documents added to ChromaDB!")
