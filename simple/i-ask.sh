#!/bin/bash

if [ -z "$OPENAI_API_KEY" ]; then
    echo "WARNING: OPENAI_API_KEY environment variable is not set."
    exit 1
fi

if [ "$(uname)" == "Darwin" ]; then
    # This is a Mac, so use Homebrew
    echo "Installing jq with Homebrew..."
    brew install jq
else
    # Assume this is a Linux system and use apt-get
    echo "Installing jq with apt-get..."
    sudo apt-get update
    sudo apt-get install jq
fi

OPENAI_API_KEY=$OPENAI_API_KEY

QUERY=$1

RESPONSE=$(curl -s https://api.openai.com/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -d '{
    "model": "gpt-3.5-turbo",
    "messages": [{"role": "system", "content": "'"$QUERY"'"}]
  }')

CONTENT=$(echo $RESPONSE | jq -r '.choices[0].message.content')

 
header="\xF0\x9F\x98\x84"
output="${header}\n\n$CONTENT\n\n"

 
echo -e "\n" "$output"