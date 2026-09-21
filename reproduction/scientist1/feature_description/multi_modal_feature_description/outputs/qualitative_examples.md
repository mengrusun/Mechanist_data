# Qualitative examples


## resnet50_layer4 — 40 sampled components

| Comp | Text label (best class match) | cos | Top-K dominant labels (count) | Class purity |
|---:|---|---:|---|---:|
| 5 | little blue heron | 0.328 | little blue heron (4), hand-held computer (3), American egret (2) | 0.44 |
| 33 | coucal | 0.373 | coucal (8), coffee mug (1) | 0.89 |
| 45 | upright | 0.326 | hair spray (2), harmonica (1), miniskirt (1) | 0.22 |
| 68 | spatula | 0.314 | spatula (4), hammer (1), cowboy hat (1) | 0.44 |
| 82 | clumber | 0.304 | mountain tent (1), viaduct (1), sax (1) | 0.11 |
| 151 | clumber | 0.318 | car wheel (1), bolo tie (1), groom (1) | 0.11 |
| 164 | jinrikisha | 0.325 | bow (2), pug (1), vine snake (1) | 0.22 |
| 182 | lumbermill | 0.304 | lumbermill (4), restaurant (1), apiary (1) | 0.44 |
| 353 | web site | 0.315 | toaster (1), seat belt (1), American lobster (1) | 0.11 |
| 358 | vestment | 0.356 | vestment (5), tusker (1), prayer rug (1) | 0.56 |
| 542 | maillot | 0.313 | conch (2), parking meter (1), stopwatch (1) | 0.22 |
| 562 | wing | 0.314 | wing (3), gown (1), tub (1) | 0.33 |
| 613 | soccer ball | 0.316 | soccer ball (3), boxer (1), Norwegian elkhound (1) | 0.33 |
| 619 | limpkin | 0.358 | limpkin (7), strainer (1), medicine chest (1) | 0.78 |
| 801 | tiger shark | 0.306 | great white shark (4), computer keyboard (2), hammerhead (1) | 0.44 |
| 865 | ashcan | 0.313 | bakery (1), Crock Pot (1), carbonara (1) | 0.11 |
| 984 | malamute | 0.324 | Eskimo dog (2), Siberian husky (2), cloak (1) | 0.22 |
| 1017 | Irish setter | 0.318 | Irish setter (4), butcher shop (1), Doberman (1) | 0.44 |
| 1027 | ashcan | 0.318 | cup (1), chime (1), disk brake (1) | 0.11 |
| 1101 | puffer | 0.316 | brain coral (4), planetarium (1), flatworm (1) | 0.44 |
| 1106 | clumber | 0.306 | basketball (2), parachute (2), jinrikisha (2) | 0.22 |
| 1128 | mortarboard | 0.322 | airship (3), mortarboard (2), cowboy hat (1) | 0.33 |
| 1134 | balance beam | 0.311 | dining table (1), bannister (1), shoji (1) | 0.11 |
| 1226 | clumber | 0.336 | printer (4), soap dispenser (1), wombat (1) | 0.44 |
| 1280 | basset | 0.319 | basset (3), bloodhound (2), suspension bridge (2) | 0.33 |
| 1311 | toy terrier | 0.340 | toy terrier (3), stopwatch (1), punching bag (1) | 0.33 |
| 1362 | baseball | 0.349 | baseball (6), cliff (1), toilet seat (1) | 0.67 |
| 1476 | patas | 0.315 | window shade (1), American Staffordshire terrier (1), chain mail (1) | 0.11 |
| 1487 | jinrikisha | 0.300 | kimono (1), lawn mower (1), ping-pong ball (1) | 0.11 |
| 1558 | clumber | 0.306 | trench coat (2), schipperke (1), miniskirt (1) | 0.22 |
| 1641 | banjo | 0.309 | acoustic guitar (4), Sussex spaniel (3), cowboy hat (1) | 0.44 |
| 1656 | chime | 0.315 | ladle (2), schooner (1), thimble (1) | 0.22 |
| 1708 | clumber | 0.310 | marmot (1), Great Dane (1), Labrador retriever (1) | 0.11 |
| 1726 | dishwasher | 0.331 | dishwasher (5), plate rack (1), lemon (1) | 0.56 |
| 1743 | banjo | 0.343 | banjo (8), drum (1) | 0.89 |
| 1762 | cash machine | 0.338 | cash machine (3), sliding door (1), teddy (1) | 0.33 |
| 1843 | lycaenid | 0.303 | overskirt (1), alligator lizard (1), plastic bag (1) | 0.11 |
| 1896 | lycaenid | 0.323 | ringlet (4), CD player (1), nipple (1) | 0.44 |
| 1963 | clumber | 0.314 | stage (1), clog (1), cuirass (1) | 0.11 |
| 2024 | web site | 0.305 | corkscrew (1), computer keyboard (1), flagpole (1) | 0.11 |

## resnet50_layer4 — text query -> nearest component

| Query | Top-1 comp | cos | Top-K image labels (dominant) |
|---|---:|---:|---|
| "goldfish" | 1092 | 0.334 | goldfish (4), carousel (1), orange (1) |
| "school bus" | 590 | 0.323 | minivan (1), ambulance (1), streetcar (1) |
| "dog" | 1221 | 0.321 | Bedlington terrier (2), wire-haired fox terrier (1), marmot (1) |
| "cat" | 1974 | 0.302 | Persian cat (4), tabby (1), four-poster (1) |
| "wheel" | 1799 | 0.307 | safety pin (1), water tower (1), barrel (1) |
| "mushroom" | 206 | 0.338 | bolete (6), agaric (1), mushroom (1) |
| "guitar" | 1158 | 0.315 | pick (8), notebook (1) |
| "keyboard" | 796 | 0.297 | Border collie (1), American alligator (1), hand-held computer (1) |
| "clock" | 1867 | 0.330 | chickadee (3), wall clock (2), analog clock (2) |
| "eye" | 892 | 0.278 | dung beetle (4), spindle (1), corn (1) |
| "text" | 178 | 0.337 | book jacket (2), perfume (2), hay (1) |
| "green field" | 690 | 0.305 | rapeseed (9) |
| "snow" | 1274 | 0.304 | snowplow (2), Rhodesian ridgeback (1), barrow (1) |
| "beach" | 160 | 0.316 | leatherback turtle (2), seashore (2), promontory (1) |
| "fire" | 1739 | 0.315 | stove (3), matchstick (3), iron (1) |

## vit_b16_cls — 40 sampled components

| Comp | Text label (best class match) | cos | Top-K dominant labels (count) | Class purity |
|---:|---|---:|---|---:|
| 2 | motor scooter | 0.319 | moped (2), worm fence (1), golfcart (1) | 0.22 |
| 12 | shoji | 0.320 | cloak (2), strainer (1), shower cap (1) | 0.22 |
| 16 | baboon | 0.334 | baboon (5), forklift (2), patas (1) | 0.56 |
| 25 | ashcan | 0.315 | whiskey jug (3), wardrobe (1), bassoon (1) | 0.33 |
| 30 | clumber | 0.311 | wine bottle (1), potter's wheel (1), crossword puzzle (1) | 0.11 |
| 55 | patas | 0.329 | radiator (1), dingo (1), Shih-Tzu (1) | 0.11 |
| 61 | jinrikisha | 0.313 | chiffonier (2), red-breasted merganser (2), wooden spoon (1) | 0.22 |
| 67 | clumber | 0.333 | teapot (1), scuba diver (1), mitten (1) | 0.11 |
| 129 | clumber | 0.316 | gorilla (2), iron (2), gibbon (1) | 0.22 |
| 133 | borzoi | 0.337 | borzoi (3), whippet (2), pot (1) | 0.33 |
| 197 | overskirt | 0.326 | apron (2), Brabancon griffon (1), aircraft carrier (1) | 0.22 |
| 207 | clumber | 0.323 | china cabinet (2), mushroom (1), boxer (1) | 0.22 |
| 225 | patas | 0.330 | komondor (3), porcupine (2), prayer rug (1) | 0.33 |
| 229 | clumber | 0.325 | cannon (3), magnetic compass (1), Lakeland terrier (1) | 0.33 |
| 296 | patas | 0.328 | whistle (1), electric locomotive (1), curly-coated retriever (1) | 0.11 |
| 324 | clumber | 0.328 | Afghan hound (1), Irish water spaniel (1), chain (1) | 0.11 |
| 368 | jinrikisha | 0.327 | bullet train (3), bell cote (1), chain (1) | 0.33 |
| 373 | ashcan | 0.323 | broom (1), eggnog (1), ear (1) | 0.11 |
| 405 | cradle | 0.322 | bassinet (1), prayer rug (1), rule (1) | 0.11 |
| 413 | clumber | 0.328 | thatch (2), tub (2), bolete (1) | 0.22 |
| 418 | sidewinder | 0.330 | king snake (1), Indian cobra (1), matchstick (1) | 0.11 |
| 450 | clumber | 0.333 | minivan (2), chain mail (1), pitcher (1) | 0.22 |
| 464 | clumber | 0.319 | rocking chair (1), king snake (1), dining table (1) | 0.11 |
| 471 | patas | 0.307 | magnetic compass (2), Chihuahua (2), kelpie (1) | 0.22 |
| 479 | clumber | 0.304 | golf ball (2), unicycle (2), spaghetti squash (1) | 0.22 |
| 503 | jinrikisha | 0.314 | shoji (2), cowboy boot (1), microphone (1) | 0.22 |
| 542 | grasshopper | 0.316 | walking stick (2), guacamole (1), maillot (1) | 0.22 |
| 553 | clumber | 0.314 | indri (2), cliff (2), Windsor tie (1) | 0.22 |
| 579 | clumber | 0.318 | fig (1), potpie (1), custard apple (1) | 0.11 |
| 600 | ashcan | 0.324 | fountain pen (2), hard disc (2), sock (1) | 0.22 |
| 611 | clumber | 0.346 | rock beauty (2), radio (1), spaghetti squash (1) | 0.22 |
| 620 | ashcan | 0.304 | tailed frog (1), projectile (1), plate (1) | 0.11 |
| 642 | clumber | 0.304 | speedboat (1), sandal (1), parachute (1) | 0.11 |
| 646 | racer | 0.320 | racer (2), ballplayer (2), stole (1) | 0.22 |
| 657 | clumber | 0.321 | bonnet (1), weasel (1), Arabian camel (1) | 0.11 |
| 675 | Pembroke | 0.321 | greenhouse (1), meat loaf (1), hoopskirt (1) | 0.11 |
| 699 | clumber | 0.308 | monitor (1), sundial (1), stage (1) | 0.11 |
| 721 | muzzle | 0.324 | saltshaker (1), pot (1), boxer (1) | 0.11 |
| 740 | clumber | 0.309 | pelican (2), chain mail (2), dishwasher (1) | 0.22 |
| 754 | lycaenid | 0.321 | platypus (1), common iguana (1), green lizard (1) | 0.11 |

## vit_b16_cls — text query -> nearest component

| Query | Top-1 comp | cos | Top-K image labels (dominant) |
|---|---:|---:|---|
| "goldfish" | 671 | 0.305 | goldfish (3), oxygen mask (1), African chameleon (1) |
| "school bus" | 507 | 0.296 | recreational vehicle (2), yurt (1), dam (1) |
| "dog" | 758 | 0.319 | dingo (3), Australian terrier (1), wild boar (1) |
| "cat" | 90 | 0.291 | Egyptian cat (3), bucket (1), mongoose (1) |
| "wheel" | 229 | 0.309 | cannon (3), magnetic compass (1), Lakeland terrier (1) |
| "mushroom" | 52 | 0.302 | envelope (2), stinkhorn (2), platypus (1) |
| "guitar" | 731 | 0.283 | flute (2), suit (1), steel drum (1) |
| "keyboard" | 456 | 0.316 | space bar (1), typewriter keyboard (1), chambered nautilus (1) |
| "clock" | 192 | 0.298 | wall clock (2), magnetic compass (1), bolo tie (1) |
| "eye" | 744 | 0.274 | face powder (2), guinea pig (1), pencil box (1) |
| "text" | 337 | 0.312 | brass (2), frying pan (1), barbershop (1) |
| "green field" | 392 | 0.286 | rapeseed (2), spaghetti squash (1), sea cucumber (1) |
| "snow" | 184 | 0.268 | water ouzel (2), school bus (2), lipstick (2) |
| "beach" | 37 | 0.291 | jersey (1), handkerchief (1), redbone (1) |
| "fire" | 458 | 0.288 | grasshopper (2), stove (1), matchstick (1) |

## vit_b16_mean — 40 sampled components

| Comp | Text label (best class match) | cos | Top-K dominant labels (count) | Class purity |
|---:|---|---:|---|---:|
| 2 | clumber | 0.323 | pier (1), slot (1), worm fence (1) | 0.11 |
| 12 | shoji | 0.314 | strainer (1), French loaf (1), sorrel (1) | 0.11 |
| 16 | clumber | 0.322 | forklift (3), baboon (2), patas (1) | 0.33 |
| 25 | clumber | 0.320 | whiskey jug (1), tiger cat (1), thresher (1) | 0.11 |
| 30 | ashcan | 0.316 | crossword puzzle (2), spider monkey (1), diaper (1) | 0.22 |
| 55 | clumber | 0.325 | dingo (1), Labrador retriever (1), coral reef (1) | 0.11 |
| 61 | clumber | 0.306 | chiffonier (2), red-breasted merganser (2), nail (1) | 0.22 |
| 67 | clumber | 0.321 | rock beauty (2), vacuum (1), hand-held computer (1) | 0.22 |
| 129 | clumber | 0.318 | gorilla (2), envelope (1), apron (1) | 0.22 |
| 133 | patas | 0.334 | borzoi (2), maraca (2), pot (1) | 0.22 |
| 197 | clumber | 0.331 | Border terrier (2), feather boa (1), Brabancon griffon (1) | 0.22 |
| 207 | clumber | 0.329 | Scottish deerhound (1), Irish wolfhound (1), papillon (1) | 0.11 |
| 225 | clumber | 0.326 | komondor (3), chainlink fence (2), theater curtain (1) | 0.33 |
| 229 | clumber | 0.318 | refrigerator (2), cannon (2), lipstick (1) | 0.22 |
| 296 | clumber | 0.315 | thunder snake (1), fly (1), whistle (1) | 0.11 |
| 324 | clumber | 0.310 | whippet (1), loggerhead (1), projectile (1) | 0.11 |
| 368 | bullet train | 0.334 | bullet train (5), bell cote (1), can opener (1) | 0.56 |
| 373 | clumber | 0.307 | matchstick (1), ear (1), broom (1) | 0.11 |
| 405 | ashcan | 0.322 | head cabbage (1), Dungeness crab (1), barometer (1) | 0.11 |
| 413 | clumber | 0.334 | thatch (3), carton (1), crash helmet (1) | 0.33 |
| 418 | buckeye | 0.325 | diamondback (2), bakery (1), banded gecko (1) | 0.22 |
| 450 | minibus | 0.333 | minibus (2), minivan (2), recreational vehicle (1) | 0.22 |
| 464 | Blenheim spaniel | 0.311 | English springer (2), mailbag (1), dining table (1) | 0.22 |
| 471 | patas | 0.315 | magnetic compass (2), Chihuahua (2), kelpie (1) | 0.22 |
| 479 | clumber | 0.317 | dingo (1), spaghetti squash (1), mitten (1) | 0.11 |
| 503 | jinrikisha | 0.321 | kimono (2), shoji (2), cowboy boot (1) | 0.22 |
| 542 | clumber | 0.320 | guacamole (1), maillot (1), American lobster (1) | 0.11 |
| 553 | lycaenid | 0.307 | indri (3), catamaran (1), Windsor tie (1) | 0.33 |
| 579 | cardoon | 0.325 | cardoon (2), fig (1), potpie (1) | 0.22 |
| 600 | hard disc | 0.334 | hard disc (3), corn (1), golfcart (1) | 0.33 |
| 611 | lycaenid | 0.328 | rock beauty (3), sewing machine (1), spaghetti squash (1) | 0.33 |
| 620 | jinrikisha | 0.317 | bathing cap (2), tailed frog (1), projectile (1) | 0.22 |
| 642 | lycaenid | 0.324 | triumphal arch (2), Scottish deerhound (1), neck brace (1) | 0.22 |
| 646 | racer | 0.353 | racer (2), switch (1), sports car (1) | 0.22 |
| 657 | clumber | 0.327 | cellular telephone (1), pizza (1), tarantula (1) | 0.11 |
| 675 | clumber | 0.308 | greenhouse (2), meat loaf (1), plow (1) | 0.22 |
| 699 | Pembroke | 0.312 | trimaran (1), malamute (1), home theater (1) | 0.11 |
| 721 | clumber | 0.304 | artichoke (1), boxer (1), grocery store (1) | 0.11 |
| 740 | Pembroke | 0.308 | pelican (3), chain mail (2), dough (1) | 0.33 |
| 754 | clumber | 0.316 | dugong (1), ptarmigan (1), platypus (1) | 0.11 |

## vit_b16_mean — text query -> nearest component

| Query | Top-1 comp | cos | Top-K image labels (dominant) |
|---|---:|---:|---|
| "goldfish" | 80 | 0.296 | jinrikisha (2), goldfish (2), axolotl (1) |
| "school bus" | 91 | 0.299 | car wheel (2), tractor (1), school bus (1) |
| "dog" | 391 | 0.315 | komondor (4), Bedlington terrier (1), Pembroke (1) |
| "cat" | 90 | 0.311 | Egyptian cat (3), tiger (2), sturgeon (1) |
| "wheel" | 72 | 0.306 | beer bottle (2), liner (1), paddlewheel (1) |
| "mushroom" | 359 | 0.313 | earthstar (6), thatch (1), gorilla (1) |
| "guitar" | 428 | 0.281 | cello (2), scabbard (1), zucchini (1) |
| "keyboard" | 456 | 0.299 | space bar (1), cassette player (1), spindle (1) |
| "clock" | 647 | 0.296 | sundial (3), cinema (1), measuring cup (1) |
| "eye" | 722 | 0.275 | triumphal arch (2), rubber eraser (1), mask (1) |
| "text" | 337 | 0.320 | brass (2), minivan (2), wig (1) |
| "green field" | 660 | 0.289 | hay (2), shopping cart (1), spotlight (1) |
| "snow" | 364 | 0.283 | wing (2), snowmobile (2), scorpion (2) |
| "beach" | 71 | 0.277 | catamaran (3), fireboat (2), pier (2) |
| "fire" | 681 | 0.285 | torch (2), sea snake (1), half track (1) |