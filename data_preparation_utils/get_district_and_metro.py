from geopy import Yandex
import pandas as pd
from geopy.distance import distance as get_distance


METRO_COUNT_RADIUS = 1.5  # km
YANDEX_API_KEY = ''


def get_district(location_obj):
    return location_obj.address.replace(', Москва, Россия', '').replace('Москва, Россия', '')


def get_nearest_metro_distances(latitude:int, longitude: int, location_objs):
    distances = []
    for obj in location_objs:
        distance_to_metro = get_distance((obj.latitude, obj.longitude), (latitude, longitude))
        distances.append(distance_to_metro)
    return sorted(distances)


if __name__ == '__main__':
    df = pd.read_csv('sales_transformed_with_coords_metro.csv')
    geolocator = Yandex(
        api_key=YANDEX_API_KEY, 
        timeout=5,
    )
    location_df = pd.DataFrame([], columns = ['district', 'near_metro_distance', 'near_metro_count'], index=addresses)

    for i, address in enumerate(addresses[180:]):
        latitude, longitude = df.loc[address][['latitude', 'longitude']].iloc[0]
        location_obj = geolocator.reverse((latitude, longitude), exactly_one=True, kind='district')
        if location_obj:
            location_df.loc[address]['district'] = get_district(location_obj)
        
        
        location_objs = geolocator.reverse((latitude, longitude), kind='metro')
        nearest_metro_distances = get_nearest_metro_distances(latitude, longitude, location_objs)
        metros_in_radius = list(filter(lambda d: d <= METRO_COUNT_RADIUS, nearest_metro_distances))
        df.at[address, 'near_metro_distance'] = nearest_metro_distances[0]    
        df.at[address, 'near_metro_count'] = len(metros_in_radius)

        if i % 10 == 0:
            print(i)

    df['near_metro_distance'] = df['near_metro_distance'].apply(lambda x: x.km if not pd.isnull(x) else None)

    df =  df.reset_index()
    df = df.drop('index', axis=1)
    df.to_csv('sales_transformed_with_coords_metro.csv')
