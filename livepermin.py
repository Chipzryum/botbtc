import requests
import datetime
import time
import pytz

def fetch_historical_data(symbol, hours=3):
    """Fetch historical minute candle data for the past 3 hours and print OHLCV."""
    utc_now = datetime.datetime.utcnow()
    # Adjust start time to fetch data up to the current minute
    start = utc_now - datetime.timedelta(hours=hours)

    endpoint = f"https://api.exchange.coinbase.com/products/{symbol}-USD/candles"
    params = {
        'start': start.isoformat(),
        'end': utc_now.isoformat(),
        'granularity': 60  # 1 minute candles
    }

    print("Fetching historical data...")

    try:
        response = requests.get(endpoint, params=params, timeout=10)
        response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)

        candles = response.json()
        # The API returns candles in reverse chronological order, so reverse for chronological printing
        candles.reverse()

        sydney_tz = pytz.timezone('Australia/Sydney')

        if not candles:
            print("No historical data returned.")
            return

        for candle in candles:
            # candle format: [timestamp, low, high, open, close, volume]
            utc_time = datetime.datetime.utcfromtimestamp(candle[0]).replace(tzinfo=pytz.utc)
            sydney_time = utc_time.astimezone(sydney_tz).strftime('%Y-%m-%d %H:%M:%S')
            low = candle[1]
            high = candle[2]
            open_price = candle[3]
            close = candle[4]
            volume = candle[5]
            print(f"{sydney_time} O:{open_price:.2f}, H:{high:.2f}, L:{low:.2f}, C:{close:.2f}, V:{volume:.2f}")

    except requests.exceptions.RequestException as e:
        print(f"Error fetching historical data: {e}")
    except Exception as e:
        print(f"An unexpected error occurred while fetching historical data: {e}")


def fetch_completed_minute_candle(symbol, retries=5, delay=3):
    """Fetch the latest completed minute candle data (OHLCV) with retries."""
    utc_now = datetime.datetime.utcnow()
    # End time is the start of the current minute (end of the previous completed minute)
    end_time = utc_now.replace(second=0, microsecond=0)
    # Start time is 1 minute before the end time
    start_time = end_time - datetime.timedelta(minutes=1)

    endpoint = f"https://api.exchange.coinbase.com/products/{symbol}-USD/candles"
    params = {
        'start': start_time.isoformat(),
        'end': end_time.isoformat(),
        'granularity': 60  # 1 minute candles
    }

    for attempt in range(retries):
        try:
            response = requests.get(endpoint, params=params, timeout=10)
            response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)

            candles = response.json()
            if candles:
                 # The API returns candles in reverse chronological order
                # The first element [0] will be the latest completed candle within the time range
                return candles[0]
            else:
                # If no data is returned, wait and retry
                if attempt < retries - 1:
                    time.sleep(delay)
                else:
                    # print(f"Failed to fetch completed candle data for {start_time.isoformat()} after multiple retries.") # Removed debug
                    return None

        except requests.exceptions.RequestException as e:
            print(f"Error fetching completed minute candle on attempt {attempt + 1}: {e}")
            if attempt < retries - 1:
                time.sleep(delay)
            else:
                print("Max retries reached for fetching completed minute candle.")
                return None
        except Exception as e:
            print(f"An unexpected error occurred while fetching completed minute candle on attempt {attempt + 1}: {e}")
            if attempt < retries - 1:
                time.sleep(delay)
            else:
                print("Max retries reached for fetching completed minute candle.")
                return None

    return None


def main():
    """Main function to fetch and print historical and live BTC prices."""
    symbol = 'BTC'
    sydney_tz = pytz.timezone('Australia/Sydney')

    # Fetch and print historical data
    fetch_historical_data(symbol)

    # Fetch and print live data (completed minute candles)
    print("\nFetching live data (completed minute candles)... (Press Ctrl+C to stop)")
    last_timestamp = None # Keep track of the last fetched candle timestamp

    try:
        while True:
            # Calculate time until the start of the next minute
            now = datetime.datetime.now(sydney_tz)
            # Calculate the time for the start of the next minute
            next_minute_start = (now.replace(second=0, microsecond=0) + datetime.timedelta(minutes=1))
            # Calculate the time difference
            time_to_next_minute_start = (next_minute_start - now).total_seconds()

            # Add a small buffer to ensure the minute is definitely over
            sleep_buffer_seconds = 2 # Wait 2 seconds into the next minute
            sleep_seconds = time_to_next_minute_start + sleep_buffer_seconds

            # Ensure sleep_seconds is not negative (shouldn't happen with this logic, but good practice)
            if sleep_seconds < 0:
                 sleep_seconds = 0

            # Wait until the start of the next minute plus the buffer
            time.sleep(sleep_seconds)

            # After waiting, the previous minute is now complete.
            # Fetch and print the OHLCV for the completed minute.
            completed_candle = fetch_completed_minute_candle(symbol)

            if completed_candle:
                # candle format: [timestamp, low, high, open, close, volume]
                utc_timestamp = completed_candle[0]

                # Only print if this is a new candle (timestamp is different from the last one)
                if utc_timestamp != last_timestamp:
                    low = completed_candle[1]
                    high = completed_candle[2]
                    open_price = completed_candle[3]
                    close = completed_candle[4]
                    volume = completed_candle[5]

                    # Convert the candle timestamp (UTC) to Sydney time for printing
                    utc_time = datetime.datetime.utcfromtimestamp(utc_timestamp).replace(tzinfo=pytz.utc)
                    sydney_time = utc_time.astimezone(sydney_tz).strftime('%Y-%m-%d %H:%M:%S')

                    # Print the full OHLCV from the completed candle
                    print(f"{sydney_time} O:{open_price:.2f}, H:{high:.2f}, L:{low:.2f}, C:{close:.2f}, V:{volume:.2f}")

                    last_timestamp = utc_timestamp # Update the last fetched timestamp
                # else: The latest completed candle is the same as the one we just printed. Do nothing.

            # The loop will naturally wait for the next minute's completion

    except KeyboardInterrupt:
        print("\nLive data fetching stopped by user.")
    except Exception as e:
        print(f"An unexpected error occurred during live data fetching: {e}")


if __name__ == '__main__':
    main()
