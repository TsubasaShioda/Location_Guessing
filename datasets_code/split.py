import os
import shutil

# 元の画像ファイルが格納されているディレクトリ
base_dir = '/Users/shiodatsubasa/データマイニング班/group/compressed_dataset'

# 新しいフォルダを生成するベースディレクトリ
output_base_dir = '/Users/shiodatsubasa/データマイニング班/group/organized_dataset'
os.makedirs(output_base_dir, exist_ok=True)

# 国と地域を対応させる辞書（重複がないように定義）
country_region_map = {
    'Aland': 'Europe',
    'Albania': 'Europe',
    'American Samoa': 'North America',
    'Andorra': 'Europe',
    'Antarctica': None,  # 地域に含めない
    'Argentina': 'South America',
    'Armenia': 'Asia（中東・日本除く）',
    'Australia': 'Oceania',
    'Austria': 'Europe',
    'Bangladesh': 'Asia（中東・日本除く）',
    'Belarus': 'Europe',
    'Belgium': 'Europe',
    'Bermuda': 'North America',
    'Bhutan': 'Asia（中東・日本除く）',
    'Bolivia': 'South America',
    'Botswana': 'Africa',
    'Brazil': 'South America',
    'Bulgaria': 'Europe',
    'Cambodia': 'Asia（中東・日本除く）',
    'Canada': 'North America',
    'Chile': 'South America',
    'China': 'Asia（中東・日本除く）',
    'Colombia': 'South America',
    'Costa Rica': 'North America',
    'Croatia': 'Europe',
    'Curacao': 'North America',
    'Czechia': 'Europe',
    'Denmark': 'Europe',
    'Dominican Republic': 'North America',
    'Ecuador': 'South America',
    'Egypt': 'Africa',
    'Estonia': 'Europe',
    'Eswatini': 'Africa',
    'Faroe Islands': 'Europe',
    'Finland': 'Europe',
    'France': 'Europe',
    'Germany': 'Europe',
    'Ghana': 'Africa',
    'Gibraltar': 'Europe',
    'Greece': 'Europe',
    'Greenland': 'North America',
    'Guam': 'North America',
    'Guatemala': 'North America',
    'Hong Kong': 'Asia（中東・日本除く）',
    'Hungary': 'Europe',
    'Iceland': 'Europe',
    'India': 'Asia（中東・日本除く）',
    'Indonesia': 'Asia（中東・日本除く）',
    'Iraq': 'Middle East',
    'Ireland': 'Europe',
    'Isle of Man': 'Europe',
    'Israel': 'Middle East',
    'Italy': 'Europe',
    'Japan': 'Japan',
    'Jersey': 'Europe',
    'Jordan': 'Middle East',
    'Kenya': 'Africa',
    'Kyrgyzstan': 'Asia（中東・日本除く）',
    'Laos': 'Asia（中東・日本除く）',
    'Latvia': 'Europe',
    'Lebanon': 'Middle East',
    'Lesotho': 'Africa',
    'Lithuania': 'Europe',
    'Luxembourg': 'Europe',
    'Macao': 'Asia（中東・日本除く）',
    'Madagascar': 'Africa',
    'Malaysia': 'Asia（中東・日本除く）',
    'Malta': 'Europe',
    'Martinique': 'North America',
    'Mexico': 'North America',
    'Monaco': 'Europe',
    'Mongolia': 'Asia（中東・日本除く）',
    'Montenegro': 'Europe',
    'Mozambique': 'Africa',
    'Myanmar': 'Asia（中東・日本除く）',
    'Nepal': 'Asia（中東・日本除く）',
    'Netherlands': 'Europe',
    'New Zealand': 'Oceania',
    'Nigeria': 'Africa',
    'North Macedonia': 'Europe',
    'Northern Mariana Islands': 'North America',
    'Norway': 'Europe',
    'Pakistan': 'Asia（中東・日本除く）',
    'Palestine': 'Middle East',
    'Paraguay': 'South America',
    'Peru': 'South America',
    'Philippines': 'Asia（中東・日本除く）',
    'Pitcairn Islands': 'Oceania',
    'Poland': 'Europe',
    'Portugal': 'Europe',
    'Puerto Rico': 'North America',
    'Qatar': 'Middle East',
    'Reunion': 'Africa',
    'Romania': 'Europe',
    'Russia': 'Europe',  # 一部アジアに属するがヨーロッパとして扱う
    'San Marino': 'Europe',
    'Senegal': 'Africa',
    'Serbia': 'Europe',
    'Singapore': 'Asia（中東・日本除く）',
    'Slovakia': 'Europe',
    'Slovenia': 'Europe',
    'South Africa': 'Africa',
    'South Georgia and South Sandwich Islands': None, # 地域に含めない
    'South Korea': 'Asia（中東・日本除く）',
    'South Sudan': 'Africa',
    'Spain': 'Europe',
    'Sri Lanka': 'Asia（中東・日本除く）',
    'Svalbard and Jan Mayen': 'Europe',
    'Sweden': 'Europe',
    'Switzerland': 'Europe',
    'Taiwan': 'Asia（中東・日本除く）',
    'Tanzania': 'Africa',
    'Thailand': 'Asia（中東・日本除く）',
    'Tunisia': 'Africa',
    'Turkey': 'Middle East',  
    'US Virgin Islands': 'North America',
    'Uganda': 'Africa',
    'Ukraine': 'Europe',
    'United Arab Emirates': 'Middle East',
    'United Kingdom': 'Europe',
    'United States': 'North America',
    'Uruguay': 'South America',
    'Venezuela': 'South America',
    'Vietnam': 'Asia（中東・日本除く）'
}

# 地域ごとのフォルダを作成
regions = set(country_region_map.values())
if None in regions:
    regions.remove(None)

for region in regions:
    if region:
        region_output_dir = os.path.join(output_base_dir, region)
        os.makedirs(region_output_dir, exist_ok=True)

# 国ごとのディレクトリを巡回してファイルをコピー
for country, region in country_region_map.items():
    if region:
        country_dir = os.path.join(base_dir, country)
        if os.path.exists(country_dir) and os.path.isdir(country_dir):
            region_output_dir = os.path.join(output_base_dir, region)
            for filename in os.listdir(country_dir):
                if filename.endswith(('.jpg', '.jpeg', '.png', '.gif')):
                    source_path = os.path.join(country_dir, filename)
                    destination_path = os.path.join(region_output_dir, filename)
                    shutil.copy(source_path, destination_path)

print("画像の地域ごとの分割（重複なしコピー）が完了しました。")