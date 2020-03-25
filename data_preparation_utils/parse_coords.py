import pandas as pd
from geopy.geocoders import Yandex
from geopy.extra.rate_limiter import RateLimiter


YANDEX_API_KEY = ''


def get_latitude(location):
    return location and location.latitude


def get_longitude(location):
    return location and location.longitude


if __name__ == '__main__':
    df = pd.read_csv('sales_transformed.csv')
    geolocator = Yandex(
        api_key=YANDEX_API_KEY, 
        timeout=5,
    )
    address_location = dict()
    for i, address in enumerate(df['address'].unique()):
        location = geolocator.geocode(address)
        address_location[address] = location
        if not location:
            print('No location {}'.format(address))
    assert set(address_location.keys()) == set(df['address'].unique()), True

    df['location'] = df['address'].apply(lambda a: address_location[a])
    df['latitude'] = df['location'].apply(get_latitude)
    df['longitude'] = df['location'].apply(get_longitude)
    df = df.drop('location', axis=1)

    df.to_csv('sales_transformed_with_coords.csv')
