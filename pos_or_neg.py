import pickle

# Dictionary of our estimates of the sentiment of the song (Måske middelværdi???)
pos_neg_dict = {
    '22': 1,
    'Afterglow': 1,
    'All_Too_Well': 0,
    'All_You_Had_To_Do_Was_Stay': 1,
    'Anti_Hero': 0,
    'August': 1,
    'Back_To_December': 0,
    'Bad_Blood': 1,
    'Begin_Again': 0,
    'Better_Than_Revenge': 0,
    'Betty': 0,
    'Blank_Space': 1,
    'Call_It_What_You_Want': 1,
    'Cardigan': 0,
    'Clean': 1,
    'Come_Back_Be_Here': 0,
    'Cornelia_Street': 1,
    'Cruel_Summer': 1,
    'Dancing_With_Our_Hands_Tied': 1,
    'Daylight': 1,
    'Dear_John': 0,
    'Death_By_A_Thousand_Cuts': 1,
    'Delicate': 1,
    'Dont_Blame_Me': 0,
    'Dress': 1,
    'Enchanted': 1,
    'Epiphany': 0,
    'False_God': 0,
    'Getaway_Car': 1,
    'Girl_At_Home': 1,
    'Gorgeous': 1,
    'Happiness': 0,
    'Haunted': 0,
    'Hoax': 0,
    'Holy_Ground': 1,
    'How_You_Get_The_Girl': 1,
    'I_Almost_Do': 1,
    'I_Can_Do_It_With_A_Broken_Heart': 1,
    'I_Did_Something_Bad': 0,
    'I_Forgot_That_You_Existed': 1,
    'I_Knew_You_Were_Trouble': 0,
    'I_Know_Places': 1,
    'I_Think_He_Knows': 1,
    'I_Wish_You_Would': 1,
    'Illicit_Affairs': 0,
    'Innocent': 0,
    'Invisible_String': 1,
    'Its_Nice_To_Have_A_Friend': 1,
    'King_Of_My_Heart': 1,
    'Last_Kiss': 0,
    'London_Boy': 1,
    'Long_Live': 1,
    'Look_What_You_Made_Me_Do': 0,
    'Lover': 1,
    'Mad_Woman': 0,
    'Me!': 1,
    'Mean': 1,
    'Mirrorball': 1,
    'Miss_Americana_And_The_Heartbreak_Prince': 1,
    'My_Tears_Ricochet': 0,
    'Never_Grow_Up': 0,
    'New_Romantics': 1,
    'New_Years_Day': 0,
    'Out_Of_The_Woods': 1,
    'Paper_Rings': 1,
    'Peace': 0,
    'Ready_For_It': 0,
    'Red': 0,
    'Right_Where_You_Left_Me': 0,
    'Sad_Beautiful_Tragic': 0,
    'Seven': 0,
    'Shake_It_Off': 1,
    'So_it_Goes': 1,
    'Sparks_Fly': 1,
    'Speak_Now': 1,
    'Starlight': 1,
    'State_Of_Grace': 1,
    'Stay_Stay_Stay': 1,
    'Style': 1,
    'The_1': 0,
    'The_Archer': 0,
    'The_Last_Great_American_Dynasty': 0,
    'The_Lucky_One': 0,
    'The_Man': 1,
    'The_Moment_I_Knew': 0,
    'The_Story_Of_Us': 0,
    'This_Is_Me_Trying': 0,
    'This_Is_Why_We_Cant_Have_Nice_Things': 1,
    'This_Love': 1,
    'Today_Was_A_Fairytale': 1,
    'Treacherous': 1,
    'We_Are_Never_Ever_Getting_Back_Together': 1,
    'Welcome_To_New_York': 1,
    'Wildest_Dreams': 1,
    'Willow': 1,
    'Wonderland': 1,
    'You_Are_In_Love': 1,
    'You_Belong_With_Me': 1,
    'You_Need_To_Calm_Down': 1,
    'A_Perfectly_Good_Heart': 0,
    'A_Place_In_This_World': 0,
    'Bejeweled': 1,
    'Bigger_Than_The_Whole_Sky': 0,
    'Breathe': 0,
    'But_Daddy_I_Love_Him': 1,
    'Cassandra': 0,
    'Champagne_Problems': 0,
    'Change': 1,
    'Chloe_Or_Sam_Or_Sophie_Or_Marcus': 0,
    'Clara_Bow': 1,
    'Closure': 0,
    'Cold_As_You': 0,
    'Come_In_With_The_Rain': 0,
    'Coney_Island': 0,
    'Cowboy_Like_Me': 0,
    'Dear_Reader': 0,
    'Dorothea': 0,
    'Down_Bad': 0,
    'Evermore': 0,
    'Everything_Has_Changed': 1,
    'Exile': 0,
    'Fearless': 1,
    'Fifteen': 1,
    'Florida': 0,
    'Forever_And_Always': 1,
    'Fortnight': 1,
    'Fresh_Out_The_Slammer': 1,
    'Glitch': 1,
    'Gold_Rush': 1,
    'Guilty_As_Sin': 1,
    'Hey_Stephen': 1,
    'High_Infidelity': 0,
    'Hits_Different': 0,
    'How_Did_It_End': 0,
    'I_Can_Fix_Him': 0,
    'I_Hate_It_Here': 0,
    'I_Look_In_Peoples_Windows': 0,
    'If_This_Was_A_Movie': 1,
    'Im_Gonna_Get_You_Back': 1,
    'Im_Only_Me_When_Im_With_You': 1,
    'Invisible': 0,
    'Its_Time_To_Go': 0,
    'Ivy': 0,
    'Jump_Then_Fall': 1,
    'Karma': 1,
    'Labyrinth': 1,
    'Lavender_Haze': 1,
    'Loml': 0,
    'Long_Story_Short': 1,
    'Love_Story': 1,
    'Marjorie': 1,
    'Maroon': 1,
    'Marys_Song': 1,
    'Mastermind': 1,
    'Midnight_Rain': 1,
    'Mine': 1,
    'My_Boy_Only_Breaks_His_Favorite_Toys': 1,
    'No_Body_No_Crime': 0,
    'Our_Song': 1,
    'Ours': 1,
    'Paris': 1,
    'Peter': 0,
    'Picture_To_Burn': 0,
    'Question': 1,
    'Robin': 0,
    'Shouldve_Said_No': 0,
    'Snow_On_The_Beach': 1,
    'So_High_School': 1,
    'So_Long_London': 0,
    'Soon_Youll_Get_Better': 0,
    'Stay_Beautiful': 1,
    'SuperStar': 1,
    'Superman': 1,
    'Sweet_Nothing': 1,
    'Teardrops_On_My_Guitar': 0,
    'Tell_Me_Why': 0,
    'Thank_You_Aimee': 0,
    'The_Albatross': 0,
    'The_Alchemy': 1,
    'The_Best_Day': 1,
    'The_Black_Dog': 0,
    'The_Bolter': 0,
    'The_Great_War': 0,
    'The_Lakes': 0,
    'The_Last_Time': 0,
    'The_Manuscript': 0,
    'The_Other_Side_Of_The_Door': 1,
    'The_Outside': 1,
    'The_Prophecy': 0,
    'The_Smallest_Man_Who_Ever_Lived': 0,
    'The_Tortured_Poets_Department': 0,
    'The_Way_I_Loved_You': 1,
    'Tied_Together_With_A_Smile': 1,
    'Tim_McGraw': 1,
    'Tis_The_Damn_Season': 0,
    'Tolerate_It': 0,
    'Untouchable': 1,
    'Vigilante_Shit': 1,
    'White_Horse': 0,
    'Whos_Afraid_Of_Little_Old_Me': 0,
    'Wouldve_Couldve_Shouldve': 0,
    'Youre_Not_Sorry': 0,
    'Youre_On_Your_Own_Kid': 0,
}

# Count 1s and 0s
count_ones = sum(value == 1 for value in pos_neg_dict.values())
count_zeros = sum(value == 0 for value in pos_neg_dict.values())

print(count_ones, count_zeros)

# Save the dictionary
with open("pos_neg_dict.pkl", "wb") as f:
    pickle.dump(pos_neg_dict, f)

from nltk.sentiment.vader import SentimentIntensityAnalyzer
import os
import numpy as np
import pickle

# Initialize the sentiment analyzer
sia = SentimentIntensityAnalyzer()

pos = 0
neg = 0
counter = 0

# Loop through all .txt files in the folder
for file_name in os.listdir("3 ugers/Lyrics"):
    file_path = os.path.join("3 ugers/Lyrics", file_name)
    
    # Read the lyrics from the file
    with open(file_path, "r") as file:
        text = file.read()
    
    # Extract the song name from the file name
    song_name = os.path.basename(file_path).replace(".txt", "")
    
    # Perform sentiment analysis
    scores = sia.polarity_scores(text)

    if np.ceil(scores["compound"]) == 1:
        pos += 1
    else:
        neg +=1

    if pos_neg_dict[song_name] != np.ceil(scores["compound"]):
        print(file_name, pos_neg_dict[song_name], scores["compound"])
        counter +=1

print(f"\npos: {pos}, neg: {neg}")
print(f"number of song mislabeled {counter}")

print("\n------------------------------------------------")
print("Song which are scored negative but is in major and other way around")
print("Song name, pos/neg, major/minor, VADER compund")
print("------------------------------------------------")
# Loop through all .txt files in the folder
for file_name in os.listdir("3 ugers/Lyrics"):
    file_path = os.path.join("3 ugers/Lyrics", file_name)

    # Load dictionary of features
    with open("feature_dict.pkl", "rb") as f:
        feature_dict = pickle.load(f)
    
    # Extract the song name from the file name
    song_name = os.path.basename(file_path).replace(".txt", "")

    if feature_dict[song_name][1] == 1:
        if np.ceil(pos_neg_dict[song_name]) == 0:
            print(file_name, pos_neg_dict[song_name], feature_dict[song_name][1], scores["compound"])
    if feature_dict[song_name][1] == -1:
        if np.ceil(pos_neg_dict[song_name]) == 1:
            print(file_name, pos_neg_dict[song_name], feature_dict[song_name][1], scores["compound"])

