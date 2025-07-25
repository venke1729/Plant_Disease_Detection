import streamlit as st
import tensorflow as tf
import numpy as np
from functools import lru_cache
import io
import time

# Language translations
TRANSLATIONS = {
    'English': {
        'select_language': 'Select Language',
        'home': 'Home',
        'about': 'About',
        'disease_recognition': 'Disease Recognition',
        'welcome_header': 'PLANT DISEASE RECOGNITION SYSTEM',
        'welcome_message': '''Welcome to the Plant Disease Recognition System! 🌿🔍
        Our mission is to help in identifying plant diseases efficiently. Upload an image of a plant, and our system will analyze it to detect any signs of diseases. Together, let's protect our crops and ensure a healthier harvest!''',
        'how_it_works': 'How It Works',
        'step1': '1. Upload Image: Go to the Disease Recognition page and upload an image of a plant with suspected diseases.',
        'step2': '2. Analysis: Our system will process the image using advanced algorithms to identify potential diseases.',
        'step3': '3. Results: View the results and recommendations for further action.',
        'why_choose_us': 'Why Choose Us?',
        'accuracy': '- Accuracy: Our system utilizes state-of-the-art machine learning techniques for accurate disease detection.',
        'user_friendly': '- User-Friendly: Simple and intuitive interface for seamless user experience.',
        'fast': '- Fast and Efficient: Receive results in seconds, allowing for quick decision-making.',
        'get_started': 'Get Started',
        'about_us': 'About Us',
        'upload_image': 'Upload Plant Image',
        'choose_image': 'Choose an Image:',
        'analyze_button': '🔍 Analyze Image',
        'processing': 'Processing image...',
        'analysis_complete': '✨ Analysis Complete!',
        'predicted_disease': '🎯 Predicted Disease:',
        'top_predictions': '📊 Top 5 Predictions',
        'disease_details': '📋 Disease Details',
        'symptoms': '🔍 Symptoms',
        'causes': '⚕️ Causes',
        'reasons': '❓ Reasons for Cause',
        'precautions': '⚠️ Precautions',
        'treatments': '💊 Treatments',
        'more_info': 'ℹ️ More Information',
        'error_occurred': '❌ An error occurred during prediction:',
        'try_again': 'Please try uploading a different image or contact support if the issue persists.',
        'instructions': '📝 Instructions',
        'tips': '💡 Tips for Best Results',
        'about_dataset': 'About Dataset',
        'dataset_intro': 'This dataset is recreated using offline augmentation from the original dataset. The original dataset can be found on this GitHub repo.',
        'dataset_description': 'This dataset consists of about 87K RGB images of healthy and diseased crop leaves categorized into 38 different classes. The total dataset is divided into an 80/20 ratio for training and validation sets while preserving directory structure.',
        'test_images': 'A new directory containing 33 test images was created later for prediction purposes.',
        'content': 'Content',
        'train_images': 'train (70,295 images)',
        'test_images_count': 'test (33 images)',
        'validation_images': 'validation (17,572 images)',
        'dataset_details': 'Dataset Details',
        'total_images': 'Total Images',
        'total_images_count': '87,900 RGB images',
        'number_of_classes': 'Number of Classes',
        'classes_count': '38 different plant diseases and healthy conditions',
        'image_resolution': 'Image Resolution',
        'resolution_type': 'High-quality RGB images',
        'data_split': 'Data Split',
        'training_split': 'Training: 80% (70,295 images)',
        'validation_split': 'Validation: 20% (17,572 images)',
        'test_split': 'Test: 33 images for prediction',
        'plant_categories': 'Plant Categories',
        'categories_intro': 'The dataset covers various plant species including:',
        'disease_types': 'Disease Types',
        'disease_types_intro': 'Each plant category includes:',
        'healthy_samples': 'Healthy samples',
        'disease_conditions': 'Various disease conditions',
        'disease_stages': 'Multiple disease stages',
        'disease_manifestations': 'Different disease manifestations',
        'data_augmentation': 'Data Augmentation',
        'augmentation_intro': 'The dataset has been enhanced through:',
        'rotation': 'Rotation',
        'flipping': 'Flipping',
        'color_adjustments': 'Color adjustments',
        'brightness_modifications': 'Brightness modifications',
        'contrast_variations': 'Contrast variations',
        'usage': 'Usage',
        'usage_intro': 'This dataset is suitable for:',
        'plant_classification': 'Plant disease classification',
        'ml_training': 'Machine learning model training',
        'cv_research': 'Computer vision research',
        'disease_detection': 'Agricultural disease detection',
        'educational_purposes': 'Educational purposes',
        'data_quality': 'Data Quality',
        'high_resolution': 'High-resolution images',
        'clear_symptoms': 'Clear disease symptoms',
        'well_labeled': 'Well-labeled categories',
        'consistent_quality': 'Consistent image quality',
        'professional_photography': 'Professional photography',
        'applications': 'Applications',
        'agricultural_detection': 'Agricultural disease detection',
        'health_monitoring': 'Plant health monitoring',
        'crop_protection': 'Crop protection',
        'research_education': 'Research and education',
        'automated_diagnosis': 'Automated disease diagnosis'
    },
    'Telugu': {
        'select_language': 'భాష ఎంచుకోండి',
        'home': 'హోమ్',
        'about': 'గురించి',
        'disease_recognition': 'వ్యాధి గుర్తింపు',
        'welcome_header': 'మొక్క వ్యాధి గుర్తింపు వ్యవస్థ',
        'welcome_message': '''మొక్క వ్యాధి గుర్తింపు వ్యవస్థకు స్వాగతం! 🌿🔍
        మొక్కల వ్యాధులను సమర్థవంతంగా గుర్తించడంలో సహాయపడటం మా లక్ష్యం. మొక్క యొక్క చిత్రాన్ని అప్‌లోడ్ చేయండి, మా వ్యవస్థ దానిని విశ్లేషించి వ్యాధి లక్షణాలను గుర్తిస్తుంది. కలిసి, మన పంటలను రక్షించుకుందాం!''',
        'how_it_works': 'ఇది ఎలా పనిచేస్తుంది',
        'step1': '1. చిత్రాన్ని అప్‌లోడ్ చేయండి: వ్యాధి గుర్తింపు పేజీకి వెళ్లి వ్యాధి అనుమానితమైన మొక్క చిత్రాన్ని అప్‌లోడ్ చేయండి.',
        'step2': '2. విశ్లేషణ: మా వ్యవస్థ అధునాతన అల్గారిథమ్‌ల ద్వారా చిత్రాన్ని విశ్లేషిస్తుంది.',
        'step3': '3. ఫలితాలు: ఫలితాలను మరియు తదుపరి చర్యల కోసం సిఫార్సులను చూడండి.',
        'why_choose_us': 'మమ్మల్ని ఎందుకు ఎంచుకోవాలి?',
        'accuracy': '- ఖచ్చితత్వం: మా వ్యవస్థ అత్యాధునిక మెషీన్ లెర్నింగ్ పద్ధతులను ఉపయోగిస్తుంది.',
        'user_friendly': '- వినియోగదారు స్నేహపూర్వకం: సరళమైన మరియు సులభమైన ఇంటర్‌ఫేస్.',
        'fast': '- వేగవంతం మరియు సమర్థవంతం: సెకన్లలో ఫలితాలను పొందండి.',
        'get_started': 'ప్రారంభించండి',
        'about_us': 'మా గురించి',
        'upload_image': 'మొక్క చిత్రాన్ని అప్‌లోడ్ చేయండి',
        'choose_image': 'చిత్రాన్ని ఎంచుకోండి:',
        'analyze_button': '🔍 చిత్రాన్ని విశ్లేషించండి',
        'processing': 'చిత్రం ప్రాసెస్ అవుతోంది...',
        'analysis_complete': '✨ విశ్లేషణ పూర్తయింది!',
        'predicted_disease': '🎯 అంచనా వేసిన వ్యాధి:',
        'top_predictions': '📊 టాప్ 5 అంచనాలు',
        'disease_details': '📋 వ్యాధి వివరాలు',
        'symptoms': '🔍 లక్షణాలు',
        'causes': '⚕️ కారణాలు',
        'reasons': '❓ కారణాల వివరణ',
        'precautions': '⚠️ జాగ్రత్తలు',
        'treatments': '💊 చికిత్సలు',
        'more_info': 'ℹ️ మరింత సమాచారం',
        'error_occurred': '❌ విశ్లేషణలో లోపం సంభవించింది:',
        'try_again': 'దయచేసి వేరే చిత్రాన్ని అప్‌లోడ్ చేయండి లేదా సహాయం కోసం సంప్రదించండి.',
        'instructions': '📝 సూచనలు',
        'tips': '💡 ఉత్తమ ఫలితాల కోసం చిట్కాలు',
        'about_dataset': 'డేటాసెట్ గురించి',
        'dataset_intro': 'ఈ డేటాసెట్ అసలు డేటాసెట్ నుండి ఆఫ్‌లైన్ ఆక్మెంటేషన్ ఉపయోగించి తిరిగి సృష్టించబడింది. అసలు డేటాసెట్ ఈ GitHub రిపోజిటరీలో కనుగొనవచ్చు.',
        'dataset_description': 'ఈ డేటాసెట్ 38 వేర్వేరు తరగతులుగా వర్గీకరించబడిన ఆరోగ్యకరమైన మరియు వ్యాధిగ్రస్తమైన పంట ఆకుల యొక్క 87K RGB చిత్రాలను కలిగి ఉంది. మొత్తం డేటాసెట్ డైరెక్టరీ నిర్మాణాన్ని కాపాడుతూ 80/20 నిష్పత్తిలో శిక్షణ మరియు ధ్రువీకరణ సెట్లుగా విభజించబడింది.',
        'test_images': 'అంచనా ప్రయోజనాల కోసం 33 పరీక్ష చిత్రాలతో కూడిన కొత్త డైరెక్టరీ తర్వాత సృష్టించబడింది.',
        'content': 'విషయ సూచిక',
        'train_images': 'శిక్షణ (70,295 చిత్రాలు)',
        'test_images_count': 'పరీక్ష (33 చిత్రాలు)',
        'validation_images': 'ధ్రువీకరణ (17,572 చిత్రాలు)',
        'dataset_details': 'డేటాసెట్ వివరాలు',
        'total_images': 'మొత్తం చిత్రాలు',
        'total_images_count': '87,900 RGB చిత్రాలు',
        'number_of_classes': 'తరగతుల సంఖ్య',
        'classes_count': '38 వేర్వేరు మొక్క వ్యాధులు మరియు ఆరోగ్యకరమైన పరిస్థితులు',
        'image_resolution': 'చిత్ర రిజల్యూషన్',
        'resolution_type': 'అధిక-నాణ్యత RGB చిత్రాలు',
        'data_split': 'డేటా విభజన',
        'training_split': 'శిక్షణ: 80% (70,295 చిత్రాలు)',
        'validation_split': 'ధ్రువీకరణ: 20% (17,572 చిత్రాలు)',
        'test_split': 'పరీక్ష: 33 చిత్రాలు అంచనా కోసం',
        'plant_categories': 'మొక్క వర్గాలు',
        'categories_intro': 'డేటాసెట్ క్రింది మొక్క జాతులను కవర్ చేస్తుంది:',
        'disease_types': 'వ్యాధి రకాలు',
        'disease_types_intro': 'ప్రతి మొక్క వర్గంలో ఇవి ఉన్నాయి:',
        'healthy_samples': 'ఆరోగ్యకరమైన నమూనాలు',
        'disease_conditions': 'వేర్వేరు వ్యాధి పరిస్థితులు',
        'disease_stages': 'అనేక వ్యాధి దశలు',
        'disease_manifestations': 'వేర్వేరు వ్యాధి వ్యక్తీకరణలు',
        'data_augmentation': 'డేటా ఆగ్మెంటేషన్',
        'augmentation_intro': 'డేటాసెట్ క్రింది మార్గాల ద్వారా మెరుగుపరచబడింది:',
        'rotation': 'భ్రమణం',
        'flipping': 'ఫ్లిపింగ్',
        'color_adjustments': 'రంగు సర్దుబాట్లు',
        'brightness_modifications': 'ప్రకాశవంతమైన సర్దుబాట్లు',
        'contrast_variations': 'కాంట్రాస్ట్ వైవిధ్యాలు',
        'usage': 'వినియోగం',
        'usage_intro': 'ఈ డేటాసెట్ క్రింది వాటికి అనువైనది:',
        'plant_classification': 'మొక్క వ్యాధి వర్గీకరణ',
        'ml_training': 'మెషీన్ లెర్నింగ్ మోడల్ శిక్షణ',
        'cv_research': 'కంప్యూటర్ విజన్ పరిశోధన',
        'disease_detection': 'వివచాయ నోయ్ గుర్తింపు',
        'educational_purposes': 'విద్యా ప్రయోజనాలు',
        'data_quality': 'డేటా నాణ్యత',
        'high_resolution': 'అధిక-రిజల్యూషన్ చిత్రాలు',
        'clear_symptoms': 'స్పష్టమైన వ్యాధి లక్షణాలు',
        'well_labeled': 'బాగా లేబుల్ చేసిన వర్గాలు',
        'consistent_quality': 'స్థిరమైన చిత్ర నాణ్యత',
        'professional_photography': 'తొఴిల్ముఱై పుకైప్పటమ్',
        'applications': 'అనువర్తనాలు',
        'agricultural_detection': 'వివచాయ నోయ్ గుర్తింపు',
        'health_monitoring': 'మొక్క ఆరోగ్య పర్యవేక్షణ',
        'crop_protection': 'పంట రక్షణ',
        'research_education': 'పరిశోధన మరియు విద్య',
        'automated_diagnosis': 'తానియఙ్కి నోయ్ గుర్తింపు'
    },
    'Tamil': {
        'select_language': 'மொழியை தேர்ந்தெடுக்கவும்',
        'home': 'முகப்பு',
        'about': 'பற்றி',
        'disease_recognition': 'நோய் கண்டறிதல்',
        'welcome_header': 'தாவர நோய் கண்டறியும் அமைப்பு',
        'welcome_message': '''தாவர நோய் கண்டறியும் அமைப்புக்கு வரவேற்கிறோம்! 🌿🔍
        தாவர நோய்களை திறம்பட கண்டறிவதற்கு உதவுவதே எங்கள் நோக்கம். தாவரத்தின் படத்தை பதிவேற்றம் செய்யுங்கள், எங்கள் அமைப்பு நோய் அறிகுறிகளை கண்டறியும்.''',
        'how_it_works': 'இது எப்படி செயல்படுகிறது',
        'step1': '1. படத்தை பதிவேற்றம் செய்யவும்: நோய் கண்டறியும் பக்கத்திற்கு சென்று தாவரத்தின் படத்தை பதிவேற்றம் செய்யவும்.',
        'step2': '2. பகுப்பாய்வு: எங்கள் அமைப்பு நவீன அல்காரிதம்கள் மூலம் படத்தை பகுப்பாய்வு செய்யும்.',
        'step3': '3. முடிவுகள்: முடிவுகளையும் அடுத்த நடவடிக்கைக்கான பரிந்துரைகளையும் காணலாம்.',
        'why_choose_us': 'எங்களை ஏன் தேர்ந்தெடுக்க வேண்டும்?',
        'accuracy': '- துல்லியம்: எங்கள் அமைப்பு நவீன மெஷின் லேர்னிங் தொழில்நுட்பங்களை பயன்படுத்துகிறது.',
        'user_friendly': '- பயனர் நட்பு: எளிமையான மற்றும் புரிந்துகொள்ள எளிதான இடைமுகம்.',
        'fast': '- வேகமானது மற்றும் திறமையானது: விநாடிகளில் முடிவுகளைப் பெறுங்கள்.',
        'get_started': 'தொடங்குங்கள்',
        'about_us': 'எங்களைப் பற்றி',
        'upload_image': 'தாவர படத்தை பதிவேற்றவும்',
        'choose_image': 'படத்தை தேர்ந்தெடுக்கவும்:',
        'analyze_button': '🔍 படத்தை பகுப்பாய்வு செய்',
        'processing': 'படம் செயலாக்கப்படுகிறது...',
        'analysis_complete': '✨ பகுப்பாய்வு முடிந்தது!',
        'predicted_disease': '🎯 கணிக்கப்பட்ட நோய்:',
        'top_predictions': '📊 முதல் 5 கணிப்புகள்',
        'disease_details': '📋 நோய் விவரங்கள்',
        'symptoms': '🔍 அறிகுறிகள்',
        'causes': '⚕️ காரணங்கள்',
        'reasons': '❓ காரணங்களுக்கான விளக்கம்',
        'precautions': '⚠️ முன்னெச்சரிக்கைகள்',
        'treatments': '💊 சிகிச்சைகள்',
        'more_info': 'ℹ️ மேலும் தகவல்',
        'error_occurred': '❌ பகுப்பாய்வில் பிழை ஏற்பட்டது:',
        'try_again': 'தயவுசெய்து வேறு படத்தை பதிவேற்றம் செய்யவும் அல்லது உதவிக்கு தொடர்பு கொள்ளவும்.',
        'instructions': '📝 வழிமுறைகள்',
        'tips': '💡 சிறந்த முடிவுகளுக்கான குறிப்புகள்',
        'about_dataset': 'தரவு தொகுப்பு பற்றி',
        'dataset_intro': 'இந்த தரவு தொகுப்பு அசல் தரவு தொகுப்பிலிருந்து ஆஃப்லைன் ஆக்மென்டேஷன் பயன்படுத்தி மீண்டும் உருவாக்கப்பட்டது. அசல் தரவு தொகுப்பை இந்த GitHub களஞ்சியத்தில் காணலாம்.',
        'dataset_description': 'இந்த தரவு தொகுப்பு 38 வெவ்வேறு வகைகளாக வகைப்படுத்தப்பட்ட ஆரோக்கியமான மற்றும் நோயுற்ற பயிர் இலைகளின் 87K RGB படங்களைக் கொண்டுள்ளது. மொத்த தரவு தொகுப்பு கோப்புறை கட்டமைப்பை பாதுகாத்து 80/20 விகிதத்தில் பயிற்சி மற்றும் சரிபார்ப்பு தொகுப்புகளாக பிரிக்கப்பட்டுள்ளது.',
        'test_images': 'கணிப்பு நோக்கங்களுக்காக 33 சோதனை படங்களைக் கொண்ட புதிய கோப்புறை பின்னர் உருவாக்கப்பட்டது.',
        'content': 'உள்ளடக்கம்',
        'train_images': 'பயிற்சி (70,295 படங்கள்)',
        'test_images_count': 'சோதனை (33 படங்கள்)',
        'validation_images': 'சரிபார்ப்பு (17,572 படங்கள்)',
        'dataset_details': 'தரவு தொகுப்பு விவரங்கள்',
        'total_images': 'மொத்த படங்கள்',
        'total_images_count': '87,900 RGB படங்கள்',
        'number_of_classes': 'வகைகளின் எண்ணிக்கை',
        'classes_count': '38 வெவ்வேறு தாவர நோய்கள் மற்றும் ஆரோக்கியமான நிலைகள்',
        'image_resolution': 'பட தெளிவு',
        'resolution_type': 'உயர்-தர RGB படங்கள்',
        'data_split': 'தரவு பிரிவு',
        'training_split': 'பயிற்சி: 80% (70,295 படங்கள்)',
        'validation_split': 'சரிபார்ப்பு: 20% (17,572 படங்கள்)',
        'test_split': 'சோதனை: 33 படங்கள் கணிப்புக்காக',
        'plant_categories': 'தாவர வகைகள்',
        'categories_intro': 'தரவு தொகுப்பு பின்வரும் தாவர இனங்களை உள்ளடக்கியது:',
        'disease_types': 'நோய் வகைகள்',
        'disease_types_intro': 'ஒவ்வொரு தாவர வகையிலும் உள்ளவை:',
        'healthy_samples': 'ஆரோக்கியமான மாதிரிகள்',
        'disease_conditions': 'பல்வேறு நோய் நிலைகள்',
        'disease_stages': 'பல நோய் நிலைகள்',
        'disease_manifestations': 'பல்வேறு நோய் வெளிப்பாடுகள்',
        'data_augmentation': 'தரவு மேம்படுத்தல்',
        'augmentation_intro': 'தரவு தொகுப்பு பின்வரும் முறைகளால் மேம்படுத்தப்பட்டுள்ளது:',
        'rotation': 'சுழற்சி',
        'flipping': 'புரட்டுதல்',
        'color_adjustments': 'நிற சரிசெய்தல்',
        'brightness_modifications': 'ஒளி சரிசெய்தல்',
        'contrast_variations': 'மாறுபாடு மாற்றங்கள்',
        'usage': 'பயன்பாடு',
        'usage_intro': 'இந்த தரவு தொகுப்பு பின்வரும் நோக்கங்களுக்கு ஏற்றது:',
        'plant_classification': 'தாவர நோய் வகைப்பாடு',
        'ml_training': 'பொறி கற்றல் மாதிரி பயிற்சி',
        'cv_research': 'கணினி பார்வை ஆராய்ச்சி',
        'disease_detection': 'விவசாய நோய் கண்டறிதல்',
        'educational_purposes': 'கல்வி நோக்கங்கள்',
        'data_quality': 'தரவு தரம்',
        'high_resolution': 'உயர்-தெளிவு படங்கள்',
        'clear_symptoms': 'தெளிவான நோய் அறிகுறிகள்',
        'well_labeled': 'நன்கு குறிக்கப்பட்ட வகைகள்',
        'consistent_quality': 'சீரான பட தரம்',
        'professional_photography': 'தொழில்முறை புகைப்படம்',
        'applications': 'பயன்பாடுகள்',
        'agricultural_detection': 'விவசாய நோய் கண்டறிதல்',
        'health_monitoring': 'தாவர ஆரோக்கிய கண்காணிப்பு',
        'crop_protection': 'பயிர் பாதுகாப்பு',
        'research_education': 'ஆராய்ச்சி மற்றும் கல்வி',
        'automated_diagnosis': 'தானியங்கி நோய் கண்டறிதல்'
    }
}

# Add custom CSS for animations and styling
st.markdown("""
<style>
    .stButton>button {
        width: 100%;
        margin-top: 10px;
        background-color: #4CAF50;
        color: white;
        border: none;
        padding: 10px 20px;
        border-radius: 5px;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #45a049;
        transform: scale(1.02);
    }
    .disease-detail {
        padding: 10px;
        margin: 5px 0;
        border-radius: 5px;
        background-color: #f0f2f6;
    }
    .prediction-bar {
        margin: 10px 0;
        padding: 10px;
        border-radius: 5px;
        background-color: #ffffff;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .main-header {
        color: #1f77b4;
        text-align: center;
        padding: 20px;
        background: linear-gradient(45deg, #1f77b4, #4CAF50);
        color: white;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .language-selector {
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state for language
if 'language' not in st.session_state:
    st.session_state.language = 'English'

# Sidebar
st.sidebar.title("Dashboard")

# Language selector in sidebar
selected_language = st.sidebar.selectbox(
    "Select Language / భాష ఎంచుకోండి / மொழியை தேர்ந்தெடுக்கவும்",
    ['English', 'Telugu', 'Tamil'],
    index=['English', 'Telugu', 'Tamil'].index(st.session_state.language)
)
st.session_state.language = selected_language

# Get translations for current language
t = TRANSLATIONS[st.session_state.language]

app_mode = st.sidebar.selectbox(t['select_language'], [t['home'], t['about'], t['disease_recognition']])

# Cache the model loading to prevent reloading on every prediction
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('trained_plant_disease_model.keras')

# Cache the image preprocessing function
@st.cache_data
def preprocess_image(image):
    # Convert Streamlit UploadedFile to bytes
    if hasattr(image, 'read'):  # Check if it's a file-like object
        image_bytes = image.read()
        image = io.BytesIO(image_bytes)
    
    # Load and preprocess image
    img = tf.keras.preprocessing.image.load_img(image, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(img)
    input_arr = np.array([input_arr])
    return input_arr

def model_prediction(test_image):
    # Load model from cache
    model = load_model()
    
    # Preprocess image from cache
    input_arr = preprocess_image(test_image)
    
    # Make prediction
    predictions = model.predict(input_arr)
    return np.argmax(predictions), predictions[0]

# Cache the disease details dictionary
@st.cache_data
def get_disease_details():
    return {
        'Apple___Apple_scab': {
            'symptoms': "Olive-green to brown spots on leaves, scabby lesions on fruit.",
            'causes': "Fungus Venturia inaequalis.",
            'reasons_for_cause': "Apple scab is caused by the fungus Venturia inaequalis, which thrives in cool, wet conditions. Spores are released from fallen leaves in the spring and infect new leaves and fruit.",
            'precautions': "Collect and destroy fallen leaves in the autumn to reduce the source of infection. Prune trees to improve air circulation and reduce humidity. Apply fungicides preventatively in the spring.",
            'treatments': "Fungicides, pruning, removing fallen leaves.",
            'detailed_info': "Apple scab is a fungal disease that affects apple and crabapple trees. It can cause significant damage to leaves and fruit, reducing the tree's vigor and yield."
        },
        'Apple___Black_rot': {
            'symptoms': "Brown spots on leaves, cankers on branches, rotting fruit.",
            'causes': "Fungus Diplodia seriata.",
            'reasons_for_cause': "Black rot is caused by the fungus Diplodia seriata, which enters trees through wounds or natural openings. The fungus can survive in dead wood and infected plant parts.",
            'precautions': "Avoid wounding trees during pruning or other activities. Remove dead or diseased wood promptly. Apply fungicides to protect wounds from infection.",
            'treatments': "Fungicides, pruning, removing infected wood.",
            'detailed_info': "Black rot is a fungal disease that affects apples, pears, and other fruit trees. It can cause leaf spots, cankers, and fruit rot, leading to significant crop losses."
        },
        'Apple___Cedar_apple_rust': {
            'symptoms': "Yellow-orange spots on leaves, raised lesions on fruit.",
            'causes': "Fungus Gymnosporangium juniperi-virginianae.",
            'reasons_for_cause': "Cedar apple rust is caused by the fungus Gymnosporangium juniperi-virginianae, which requires both apple and cedar trees to complete its life cycle. Spores are released from cedar galls in the spring and infect apple trees.",
            'precautions': "Remove cedar trees from the vicinity of apple trees to break the disease cycle. Apply fungicides to protect apple trees during periods of spore release.",
            'treatments': "Fungicides, removing cedar trees from proximity.",
            'detailed_info': "Cedar apple rust is a fungal disease that requires both apple and cedar trees to complete its life cycle. It causes yellow-orange spots on apple leaves and galls on cedar trees."
        },
        'Apple___healthy': {
            'symptoms': "No visible symptoms.",
            'causes': "N/A",
            'reasons_for_cause': "N/A",
            'precautions': "Maintain good tree health through proper watering, fertilization, and pruning.",
            'treatments': "N/A",
            'detailed_info': "A healthy apple tree exhibits no signs of disease or pest infestation. Leaves are green and vibrant, and fruit is free from blemishes."
        },
        'Blueberry___healthy': {
            'symptoms': "No visible symptoms.",
            'causes': "N/A",
            'reasons_for_cause': "N/A",
            'precautions': "Maintain good plant health through proper watering, fertilization, and pruning.",
            'treatments': "N/A",
            'detailed_info': "A healthy blueberry plant is characterized by vigorous growth, abundant fruit production, and the absence of disease or pest damage."
        },
        'Cherry_(including_sour)___Powdery_mildew': {
            'symptoms': "White powdery growth on leaves and fruit.",
            'causes': "Fungus Podosphaera clandestina.",
            'reasons_for_cause': "Powdery mildew is caused by the fungus Podosphaera clandestina, which thrives in humid conditions. Spores are spread by wind and infect new plant tissue.",
            'precautions': "Improve air circulation around plants by pruning and spacing them adequately. Avoid overhead watering, which can increase humidity. Apply fungicides preventatively.",
            'treatments': "Fungicides, good air circulation.",
            'detailed_info': "Powdery mildew is a fungal disease that affects a wide range of plants, including cherries. It causes a white powdery growth on leaves, stems, and fruit."
        },
        'Cherry_(including_sour)___healthy': {
            'symptoms': "No visible symptoms.",
            'causes': "N/A",
            'reasons_for_cause': "N/A",
            'precautions': "Maintain good tree health through proper watering, fertilization, and pruning.",
            'treatments': "N/A",
            'detailed_info': "A healthy cherry tree displays strong growth, dark green leaves, and abundant, high-quality fruit."
        },
        'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot': {
            'symptoms': "Grayish-brown rectangular lesions on leaves.",
            'causes': "Fungus Cercospora zeae-maydis.",
            'reasons_for_cause': "Gray leaf spot is caused by the fungus Cercospora zeae-maydis, which survives in crop residue. Spores are spread by wind and rain and infect new leaves.",
            'precautions': "Practice crop rotation to reduce the buildup of inoculum in the soil. Use resistant corn varieties. Apply fungicides preventatively.",
            'treatments': "Fungicides, crop rotation, resistant varieties.",
            'detailed_info': "Gray leaf spot is a fungal disease that affects corn. It causes grayish-brown lesions on leaves, which can reduce photosynthetic activity and yield."
        },
        'Corn_(maize)___Common_rust_': {
            'symptoms': "Reddish-brown pustules on leaves.",
            'causes': "Fungus Puccinia sorghi.",
            'reasons_for_cause': "Common rust is caused by the fungus Puccinia sorghi, which requires a alternate host. Spores are spread by wind and infect new leaves.",
            'precautions': "Use resistant corn varieties. Apply fungicides preventatively.",
            'treatments': "Fungicides, resistant varieties.",
            'detailed_info': "Common rust is a fungal disease that affects corn. It causes reddish-brown pustules on leaves and stalks, which can reduce yield and grain quality."
        },
        'Corn_(maize)___Northern_Leaf_Blight': {
            'symptoms': "Long, elliptical, gray-green lesions on leaves.",
            'causes': "Fungus Exserohilum turcicum.",
            'reasons_for_cause': "Northern leaf blight is caused by the fungus Exserohilum turcicum, which survives in crop residue. Spores are spread by wind and rain and infect new leaves.",
            'precautions': "Practice crop rotation to reduce the buildup of inoculum in the soil. Use resistant corn varieties. Apply fungicides preventatively.",
            'treatments': "Fungicides, crop rotation, resistant varieties.",
            'detailed_info': "Northern leaf blight is a fungal disease that affects corn. It causes long, elliptical lesions on leaves, which can reduce photosynthetic activity and yield."
        },
        'Corn_(maize)___healthy': {
            'symptoms': "No visible symptoms.",
            'causes': "N/A",
            'reasons_for_cause': "N/A",
            'precautions': "Maintain good plant health through proper watering, fertilization, and weed control.",
            'treatments': "N/A",
            'detailed_info': "A healthy corn plant exhibits vigorous growth, dark green leaves, and well-developed ears. It is free from disease and pest damage."
        },
        'Grape___Black_rot': {
            'symptoms': "Reddish-brown spots on leaves, black lesions on fruit.",
            'causes': "Fungus Guignardia bidwellii.",
            'reasons_for_cause': "Black rot is caused by the fungus Guignardia bidwellii, which survives in infected plant parts. Spores are spread by wind and rain and infect new leaves and fruit.",
            'precautions': "Remove and destroy infected plant parts. Apply fungicides preventatively.",
            'treatments': "Fungicides, removing infected plant parts.",
            'detailed_info': "Black rot is a fungal disease that affects grapes. It causes reddish-brown spots on leaves and black, shriveled berries."
        },
        'Grape___Esca_(Black_Measles)': {
            'symptoms': "Leaf spots, wood decay, fruit discoloration.",
            'causes': "Complex of fungi.",
            'reasons_for_cause': "Esca is caused by a complex of fungi that infect grapevines through wounds. The fungi can survive in dead wood and infected plant parts.",
            'precautions': "Avoid wounding grapevines during pruning or other activities. Remove dead or diseased wood promptly. Apply wound protectants.",
            'treatments': "Pruning, wound protection, trunk surgery.",
            'detailed_info': "Esca, also known as black measles, is a complex fungal disease that affects grapevines. It can cause leaf spots, wood decay, and fruit discoloration."
        },
        'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)': {
            'symptoms': "Small, circular spots on leaves.",
            'causes': "Fungus Isariopsis clavispora.",
            'reasons_for_cause': "Isariopsis leaf spot is caused by the fungus Isariopsis clavispora. The fungus survives in leaf litter and infected plant parts. Spores are spread by wind and rain and infect new leaves.",
            'precautions': "Remove and destroy infected leaves. Apply fungicides preventatively.",
            'treatments': "Fungicides, removing infected leaves.",
            'detailed_info': "Isariopsis leaf spot is a fungal disease that affects grapes. It causes small, circular spots on leaves, which can lead to defoliation."
        },
        'Grape___healthy': {
            'symptoms': "No visible symptoms.",
            'causes': "N/A",
            'reasons_for_cause': "N/A",
            'precautions': "Maintain good vine health through proper watering, fertilization, and pruning.",
            'treatments': "N/A",
            'detailed_info': "A healthy grape vine exhibits strong growth, dark green leaves, and abundant, high-quality fruit."
        },
        'Orange___Haunglongbing_(Citrus_greening)': {
            'symptoms': "Blotchy mottling of leaves, asymmetrical fruit.",
            'causes': "Bacterium Candidatus Liberibacter asiaticus.",
            'reasons_for_cause': "Huanglongbing (HLB) is caused by the bacterium Candidatus Liberibacter asiaticus, which is transmitted by psyllids. The bacterium infects the phloem of citrus trees, disrupting nutrient transport.",
            'precautions': "Control psyllid populations with insecticides. Remove infected trees promptly.",
            'treatments': "No cure, control psyllids, remove infected trees.",
            'detailed_info': "Huanglongbing (HLB), also known as citrus greening, is a devastating bacterial disease that affects citrus trees. It causes blotchy mottling of leaves and asymmetrical, bitter fruit."
        },
        'Peach___Bacterial_spot': {
            'symptoms': "Small, dark spots on leaves and fruit.",
            'causes': "Bacterium Xanthomonas campestris pv. pruni.",
            'reasons_for_cause': "Bacterial spot is caused by the bacterium Xanthomonas campestris pv. pruni, which enters trees through wounds or natural openings. The bacterium thrives in warm, wet conditions.",
            'precautions': "Avoid wounding trees during pruning or other activities. Apply copper-based fungicides preventatively.",
            'treatments': "Copper-based fungicides, pruning.",
            'detailed_info': "Bacterial spot is a bacterial disease that affects peaches, plums, and other stone fruit trees. It causes small, dark spots on leaves and fruit."
        },
        'Peach___healthy': {
            'symptoms': "No visible symptoms.",
            'causes': "N/A",
            'reasons_for_cause': "N/A",
            'precautions': "Maintain good tree health through proper watering, fertilization, and pruning.",
            'treatments': "N/A",
            'detailed_info': "A healthy peach tree exhibits vigorous growth, dark green leaves, and abundant, high-quality fruit."
        },
        'Pepper,_bell___Bacterial_spot': {
            'symptoms': "Dark, water-soaked spots on leaves and fruit.",
            'causes': "Bacterium Xanthomonas vesicatoria.",
            'reasons_for_cause': "Bacterial spot is caused by the bacterium Xanthomonas vesicatoria, which is spread by splashing water and contaminated seed. The bacterium thrives in warm, humid conditions.",
            'precautions': "Use disease-free seed. Avoid overhead watering. Apply copper-based fungicides preventatively.",
            'treatments': "Copper-based fungicides, removing infected plants.",
            'detailed_info': "Bacterial spot is a bacterial disease that affects peppers and tomatoes. It causes dark, water-soaked spots on leaves and fruit."
        },
        'Pepper,_bell___healthy': {
            'symptoms': "No visible symptoms.",
            'causes': "N/A",
            'reasons_for_cause': "N/A",
            'precautions': "Maintain good plant health through proper watering, fertilization, and pest control.",
            'treatments': "N/A",
            'detailed_info': "A healthy bell pepper plant displays strong growth, dark green leaves, and abundant, high-quality fruit."
        },
        'Potato___Early_blight': {
            'symptoms': "Dark, concentric spots on leaves.",
            'causes': "Fungus Alternaria solani.",
            'reasons_for_cause': "Early blight is caused by the fungus Alternaria solani, which survives in crop residue. Spores are spread by wind and rain and infect new leaves.",
            'precautions': "Practice crop rotation to reduce the buildup of inoculum in the soil. Use disease-free seed potatoes. Apply fungicides preventatively.",
            'treatments': "Fungicides, crop rotation.",
            'detailed_info': "Early blight is a fungal disease that affects potatoes and tomatoes. It causes dark, concentric spots on leaves."
        },
        'Potato___Late_blight': {
            'symptoms': "Water-soaked lesions on leaves, white mold.",
            'causes': "Oomycete Phytophthora infestans.",
            'reasons_for_cause': "Late blight is caused by the oomycete Phytophthora infestans, which spreads rapidly in cool, wet conditions. Spores are spread by wind and rain and infect new leaves and tubers.",
            'precautions': "Use disease-free seed potatoes. Avoid overhead watering. Apply fungicides preventatively.",
            'treatments': "Fungicides, removing infected plants.",
            'detailed_info': "Late blight is a devastating disease that affects potatoes and tomatoes. It causes water-soaked lesions on leaves and white mold."
        },
        'Potato___healthy': {
            'symptoms': "No visible symptoms.",
            'causes': "N/A",
            'reasons_for_cause': "N/A",
            'precautions': "Maintain good plant health through proper watering, fertilization, and pest control.",
            'treatments': "N/A",
            'detailed_info': "A healthy potato plant exhibits vigorous growth and abundant tuber production. It is free from disease and pest damage."
        },
        'Raspberry___healthy': {
            'symptoms': "No visible symptoms.",
            'causes': "N/A",
            'reasons_for_cause': "N/A",
            'precautions': "Maintain good plant health through proper watering, fertilization, and pest control.",
            'treatments': "N/A",
            'detailed_info': "A healthy raspberry plant is characterized by vigorous growth, abundant fruit production, and the absence of disease or pest damage."
        },
        'Soybean___healthy': {
            'symptoms': "No visible symptoms.",
            'causes': "N/A",
            'reasons_for_cause': "N/A",
            'precautions': "Maintain good plant health through proper watering, fertilization, and weed control.",
            'treatments': "N/A",
            'detailed_info': "A healthy soybean plant exhibits strong growth, dark green leaves, and abundant pod production. It is free from disease and pest damage."
        },
        'Squash___Powdery_mildew': {
            'symptoms': "White powdery growth on leaves and stems.",
            'causes': "Fungi (various species).",
            'reasons_for_cause': "Powdery mildew is caused by various species of fungi that thrive in humid conditions. Spores are spread by wind and infect new plant tissue.",
            'precautions': "Improve air circulation around plants by pruning and spacing them adequately. Avoid overhead watering, which can increase humidity. Apply fungicides preventatively.",
            'treatments': "Fungicides, good air circulation.",
            'detailed_info': "Powdery mildew is a fungal disease that affects a wide range of plants, including squash. It causes a white powdery growth on leaves and stems."
        },
        'Strawberry___Leaf_scorch': {
            'symptoms': "Small, purple spots on leaves that merge and darken.",
            'causes': "Fungus Diplocarpon earlianum.",
            'reasons_for_cause': "Leaf scorch is caused by the fungus Diplocarpon earlianum, which survives in infected plant parts. Spores are spread by splashing water and infect new leaves.",
            'precautions': "Remove and destroy infected leaves. Apply fungicides preventatively.",
            'treatments': "Fungicides, removing infected leaves.",
            'detailed_info': "Leaf scorch is a fungal disease that affects strawberries. It causes small, purple spots on leaves that merge and darken."
        },
        'Strawberry___healthy': {
            'symptoms': "No visible symptoms.",
            'causes': "N/A",
            'reasons_for_cause': "N/A",
            'precautions': "Maintain good plant health through proper watering, fertilization, and pest control.",
            'treatments': "N/A",
            'detailed_info': "A healthy strawberry plant displays strong growth, dark green leaves, and abundant, high-quality fruit."
        },
        'Tomato___Bacterial_spot': {
            'symptoms': "Small, dark spots on leaves and fruit.",
            'causes': "Bacterium Xanthomonas vesicatoria.",
            'reasons_for_cause': "Bacterial spot is caused by the bacterium Xanthomonas vesicatoria, which is spread by splashing water and contaminated seed. The bacterium thrives in warm, humid conditions.",
            'precautions': "Use disease-free seed. Avoid overhead watering. Apply copper-based fungicides preventatively.",
            'treatments': "Copper-based fungicides, removing infected plants.",
            'detailed_info': "Bacterial spot is a bacterial disease that affects tomatoes and peppers. It causes small, dark spots on leaves and fruit."
        },
        'Tomato___Early_blight': {
            'symptoms': "Dark, concentric spots on leaves.",
            'causes': "Fungus Alternaria solani.",
            'reasons_for_cause': "Early blight is caused by the fungus Alternaria solani, which survives in crop residue. Spores are spread by wind and rain and infect new leaves.",
            'precautions': "Practice crop rotation to reduce the buildup of inoculum in the soil. Use disease-free transplants. Apply fungicides preventatively.",
            'treatments': "Fungicides, crop rotation.",
            'detailed_info': "Early blight is a fungal disease that affects tomatoes and potatoes. It causes dark, concentric spots on leaves."
        },
        'Tomato___Late_blight': {
            'symptoms': "Water-soaked lesions on leaves, white mold.",
            'causes': "Oomycete Phytophthora infestans.",
            'reasons_for_cause': "Late blight is caused by the oomycete Phytophthora infestans, which spreads rapidly in cool, wet conditions. Spores are spread by wind and rain and infect new leaves and fruit.",
            'precautions': "Use disease-free transplants. Avoid overhead watering. Apply fungicides preventatively.",
            'treatments': "Fungicides, removing infected plants.",
            'detailed_info': "Late blight is a devastating disease that affects tomatoes and potatoes. It causes water-soaked lesions on leaves and white mold."
        },
        'Tomato___Leaf_Mold': {
            'symptoms': "Pale green or yellow spots on upper leaf surface, gray-purple mold on lower surface.",
            'causes': "Fungus Passalora fulva.",
            'reasons_for_cause': "Leaf mold is caused by the fungus Passalora fulva, which thrives in humid conditions. Spores are spread by wind and infect new leaves.",
            'precautions': "Improve air circulation around plants by pruning and spacing them adequately. Avoid overhead watering, which can increase humidity. Apply fungicides preventatively.",
            'treatments': "Fungicides, good air circulation.",
            'detailed_info': "Leaf mold is a fungal disease that affects tomatoes. It causes pale green or yellow spots on the upper leaf surface and gray-purple mold on the lower surface."
        },
        'Tomato___Septoria_leaf_spot': {
            'symptoms': "Small, circular spots with dark borders and light centers on leaves.",
            'causes': "Fungus Septoria lycopersici.",
            'reasons_for_cause': "Septoria leaf spot is caused by the fungus Septoria lycopersici, which survives in crop residue. Spores are spread by splashing water and infect new leaves.",
            'precautions': "Practice crop rotation to reduce the buildup of inoculum in the soil. Remove and destroy infected leaves. Apply fungicides preventatively.",
            'treatments': "Fungicides, removing infected leaves, crop rotation.",
            'detailed_info': "Septoria leaf spot is a fungal disease that affects tomatoes. It causes small, circular spots with dark borders and light centers on leaves."
        },
        'Tomato___Spider_mites Two-spotted_spider_mite': {
            'symptoms': "Fine webbing on leaves, yellowing or stippling.",
            'causes': "Spider mites (Tetranychus urticae).",
            'reasons_for_cause': "Spider mites are tiny pests that thrive in hot, dry conditions. They feed on plant sap, causing damage to leaves.",
            'precautions': "Maintain adequate soil moisture. Control weeds, which can serve as alternate hosts. Introduce beneficial insects, such as predatory mites.",
            'treatments': "Insecticides, miticides, biological control.",
            'detailed_info': "Spider mites are tiny pests that can infest tomatoes. They cause fine webbing on leaves and yellowing or stippling."
        },
        'Tomato___Target_Spot': {
            'symptoms': "Small, circular spots with concentric rings on leaves and fruit.",
            'causes': "Fungus Corynespora cassiicola.",
            'reasons_for_cause': "Target spot is caused by the fungus Corynespora cassiicola, which survives in crop residue. Spores are spread by wind and rain and infect new leaves and fruit.",
            'precautions': "Practice crop rotation to reduce the buildup of inoculum in the soil. Apply fungicides preventatively.",
            'treatments': "Fungicides, crop rotation.",
            'detailed_info': "Target spot is a fungal disease that affects tomatoes. It causes small, circular spots with concentric rings on leaves and fruit."
        },
        'Tomato___Tomato_Yellow_Leaf_Curl_Virus': {
            'symptoms': "Yellowing and curling of leaves, stunted growth.",
            'causes': "Tomato yellow leaf curl virus (TYLCV).",
            'reasons_for_cause': "Tomato yellow leaf curl virus (TYLCV) is transmitted by whiteflies. The virus infects tomato plants, causing yellowing and curling of leaves and stunted growth.",
            'precautions': "Control whitefly populations with insecticides. Use resistant tomato varieties. Remove infected plants promptly.",
            'treatments': "Insecticides to control whiteflies, resistant varieties.",
            'detailed_info': "Tomato yellow leaf curl virus (TYLCV) is a viral disease that affects tomatoes. It causes yellowing and curling of leaves and stunted growth."
        },
        'Tomato___Tomato_mosaic_virus': {
            'symptoms': "Mottled leaves, stunted growth, reduced fruit yield.",
            'causes': "Tomato mosaic virus (ToMV).",
            'reasons_for_cause': "Tomato mosaic virus (ToMV) is transmitted by contact. The virus infects tomato plants, causing mottled leaves, stunted growth, and reduced fruit yield.",
            'precautions': "Use disease-free transplants. Wash hands and tools thoroughly after handling tomato plants. Remove infected plants promptly.",
            'treatments': "Removing infected plants, sanitation.",
            'detailed_info': "Tomato mosaic virus (ToMV) is a viral disease that affects tomatoes. It causes mottled leaves, stunted growth, and reduced fruit yield."
        },
        'Tomato___healthy': {
            'symptoms': "No visible symptoms.",
            'causes': "N/A",
            'reasons_for_cause': "N/A",
            'precautions': "Maintain good plant health through proper watering, fertilization, and pest control.",
            'treatments': "N/A",
            'detailed_info': "A healthy tomato plant exhibits vigorous growth, dark green leaves, and abundant, high-quality fruit."
        }
    }

# Cache the class names list
@st.cache_data
def get_class_names():
    return [
        'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
        'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew',
        'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
        'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy',
        'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
        'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
        'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy',
        'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
        'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew',
        'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot',
        'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
        'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
        'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
        'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
    ]

# Main Page
if app_mode == t['home']:
    st.header(t['welcome_header'])
    image_path = "home_page.jpeg"
    st.image(image_path, use_column_width=True)
    st.markdown(t['welcome_message'])
    st.markdown(f"### {t['how_it_works']}")
    st.markdown(t['step1'])
    st.markdown(t['step2'])
    st.markdown(t['step3'])
    st.markdown(f"### {t['why_choose_us']}")
    st.markdown(t['accuracy'])
    st.markdown(t['user_friendly'])
    st.markdown(t['fast'])
    st.markdown(f"### {t['get_started']}")
    st.markdown(f"### {t['about_us']}")

# About Project
elif app_mode == t['about']:
    st.header(t['about'])
    st.markdown(f"""
    #### {t['about_dataset']}
    {t['dataset_intro']}
    
    {t['dataset_description']}
    
    {t['test_images']}
    
    #### {t['content']}
    1. {t['train_images']}
    2. {t['test_images_count']}
    3. {t['validation_images']}

    #### {t['dataset_details']}
    - **{t['total_images']}**: {t['total_images_count']}
    - **{t['number_of_classes']}**: {t['classes_count']}
    - **{t['image_resolution']}**: {t['resolution_type']}
    - **{t['data_split']}**:
        - {t['training_split']}
        - {t['validation_split']}
        - {t['test_split']}

    #### {t['plant_categories']}
    {t['categories_intro']}
    - Apple
    - Blueberry
    - Cherry
    - Corn (Maize)
    - Grape
    - Orange
    - Peach
    - Pepper
    - Potato
    - Raspberry
    - Soybean
    - Squash
    - Strawberry
    - Tomato

    #### {t['disease_types']}
    {t['disease_types_intro']}
    - {t['healthy_samples']}
    - {t['disease_conditions']}
    - {t['disease_stages']}
    - {t['disease_manifestations']}

    #### {t['data_augmentation']}
    {t['augmentation_intro']}
    - {t['rotation']}
    - {t['flipping']}
    - {t['color_adjustments']}
    - {t['brightness_modifications']}
    - {t['contrast_variations']}

    #### {t['usage']}
    {t['usage_intro']}
    - {t['plant_classification']}
    - {t['ml_training']}
    - {t['cv_research']}
    - {t['disease_detection']}
    - {t['educational_purposes']}

    #### {t['data_quality']}
    - {t['high_resolution']}
    - {t['clear_symptoms']}
    - {t['well_labeled']}
    - {t['consistent_quality']}
    - {t['professional_photography']}

    #### {t['applications']}
    - {t['agricultural_detection']}
    - {t['health_monitoring']}
    - {t['crop_protection']}
    - {t['research_education']}
    - {t['automated_diagnosis']}
    """)

# Prediction Page
elif app_mode == t['disease_recognition']:
    st.markdown('<h1 class="main-header">Plant Disease Recognition</h1>', unsafe_allow_html=True)
    
    # Create two columns with adjusted widths
    col1, col2 = st.columns([1, 1.5])
    
    with col1:
        st.subheader(t['upload_image'])
        test_image = st.file_uploader(t['choose_image'], type=['jpg', 'jpeg', 'png'])
        
        if test_image is not None:
            # Display image with reduced size and center alignment
            st.markdown('<div style="text-align: center;">', unsafe_allow_html=True)
            st.image(test_image, width=300)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Animated predict button with custom styling
            st.markdown("""
                <style>
                .stButton>button {
                    width: 100%;
                    margin-top: 10px;
                    background-color: #4CAF50;
                    color: white;
                    border: none;
                    padding: 10px 20px;
                    border-radius: 5px;
                    transition: all 0.3s ease;
                }
                .stButton>button:hover {
                    background-color: #45a049;
                    transform: scale(1.02);
                }
                </style>
            """, unsafe_allow_html=True)
            
            if st.button(t['analyze_button'], key="predict_button"):
                with st.spinner(t['processing']):
                    try:
                        # Add a small delay for animation
                        time.sleep(0.5)
                        
                        result_index, predicted_probabilities = model_prediction(test_image)
                        
                        # Get cached data
                        class_names = get_class_names()
                        disease_details = get_disease_details()

                        predicted_disease = class_names[result_index]
                        
                        # Animated success message
                        st.success(t['analysis_complete'])
                        st.markdown(f"### {t['predicted_disease']} {predicted_disease}")

                        # Display top 5 predictions with animated progress bars
                        st.markdown(f"### {t['top_predictions']}")
                        top_5_indices = np.argsort(predicted_probabilities)[-5:][::-1]
                        
                        for idx in top_5_indices:
                            prob = predicted_probabilities[idx]
                            with st.container():
                                st.markdown(f'<div class="prediction-bar">', unsafe_allow_html=True)
                                st.progress(float(prob))
                                st.write(f"**{class_names[idx]}:** {prob:.2f}")
                                st.markdown('</div>', unsafe_allow_html=True)
                                time.sleep(0.2)  # Add delay for animation

                        # Display Disease Details with bullet points
                        if predicted_disease in disease_details:
                            st.markdown(f"### {t['disease_details']}")
                            details = disease_details[predicted_disease]
                            
                            # Create expandable sections for each detail with custom styling
                            st.markdown("""
                                <style>
                                .streamlit-expanderHeader {
                                    background-color: #f0f2f6;
                                    border-radius: 5px;
                                    padding: 10px;
                                    margin: 5px 0;
                                }
                                </style>
                            """, unsafe_allow_html=True)
                            
                            # Display details in a more compact format
                            with st.expander(f"{t['symptoms']}", expanded=True):
                                st.markdown(f"• {details['symptoms']}")
                            
                            with st.expander(f"{t['causes']}", expanded=True):
                                st.markdown(f"• {details['causes']}")
                            
                            with st.expander(f"{t['reasons']}", expanded=True):
                                st.markdown(f"• {details['reasons_for_cause']}")
                            
                            with st.expander(f"{t['precautions']}", expanded=True):
                                st.markdown(f"• {details['precautions']}")
                            
                            with st.expander(f"{t['treatments']}", expanded=True):
                                st.markdown(f"• {details['treatments']}")
                            
                            with st.expander(f"{t['more_info']}", expanded=True):
                                st.markdown(f"• {details['detailed_info']}")
                        else:
                            st.warning("⚠️ Disease details not found.")
                            
                    except Exception as e:
                        st.error(f"{t['error_occurred']} {str(e)}")
                        st.error(t['try_again'])
    
    with col2:
        st.markdown(f"### {t['instructions']}")
        st.markdown(f"""
        1. {t['step1']}
        2. {t['step2']}
        3. {t['step3']}
        
        ### {t['tips']}
        - Use well-lit images
        - Ensure the leaf is clearly visible
        - Avoid blurry or dark images
        - Capture both healthy and diseased parts if possible
        """)
