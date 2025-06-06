import phonenumbers
from phonenumbers import geocoder
phone_number1 = phonenumbers.parse("+254740889802")
phone_number2 = phonenumbers.parse("+254111206173")


print("\nPhone Numbers Location\n")
print(geocoder.description_for_number(phone_number1, "en"));
print(geocoder.description_for_number(phone_number2, "en"));


from phonenumbers import carrier

print("\nCarrier Information\n")
print(carrier.name_for_number(phone_number1, "en"))
print(carrier.name_for_number(phone_number2, "en"))


import phonenumbers
number = phonenumbers.parse("+254740889802")
print(number)


from phonenumbers import timezone
timezone.time_zones_for_number(number)

import pywhatkit as pwk
phone_number = "+254740889802"
message = "Hello, this is an automated message sent using python!"
#send the message
pwk.sendwhatmsg(phone_number, message, time_hour:13, time_min:24)