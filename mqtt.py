import paho.mqtt.client as mqtt
import json

# The unique channel and public broker
MQTT_TOPIC = "farshad/s2vmo/dome/stream"
MQTT_BROKER = "broker.hivemq.com" 
MQTT_PORT = 1883

def on_connect(client, userdata, flags, reason_code, properties=None):
    if reason_code == 0:
        print(f"✅ Successfully connected to Public HiveMQ Cloud!")
        client.subscribe(MQTT_TOPIC)
        print(f"📡 Listening for dome data on: {MQTT_TOPIC}...\n")
    else:
        print(f"❌ Failed to connect. Reason code: {reason_code}")

def on_message(client, userdata, msg):
    print("--------------------------------------------------")
    print(f"📥 New S2VMO Data Received")
    try:
        payload_str = msg.payload.decode('utf-8')
        data = json.loads(payload_str)
        print(json.dumps(data, indent=4))
    except Exception as e:
        print(f"⚠️ Error processing message: {e}")

client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
client.on_connect = on_connect
client.on_message = on_message

print("Connecting to Cloud Broker...")
client.connect(MQTT_BROKER, MQTT_PORT, 60)
client.loop_forever()