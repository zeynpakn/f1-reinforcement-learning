# ── Ekran ──────────────────────────────────────────
SCREEN_WIDTH  = 1000
SCREEN_HEIGHT = 700
FPS           = 60
HEADLESS      = True # Pencere açmadan çalıştırmayı ifade ediyor, 
# trainde pencere açılmadan train yapılıyor ki hızlı olsun

# ── Araba ──────────────────────────────────────────
CAR_SPEED_MAX    = 6
CAR_ACCELERATION = 0.3
CAR_FRICTION     = 0.05
CAR_TURN_SPEED   = 3 # Her dönüşte açı ne kadar değişsin 

# ── Sensör ─────────────────────────────────────────
SENSOR_COUNT     = 5
SENSOR_MAX_RANGE = 150 # Sensör max 150 pixel uzağı görebiliyor.
SENSOR_ANGLES    = [-60, -30, 0, 30, 60]

# ── Ortam ──────────────────────────────────────────
MAX_STEPS_PER_EPISODE = 2000 
# Agent'i pisti tam tur dönmesine sınır konuldu çünkü bir yerde takılırsa sonsuz döngüde kalmasın

# ── Reward ─────────────────────────────────────────
REWARD_ALIVE        =  2 # araba düzgün gidiyorsa ödül 
REWARD_CHECKPOINT   =  50 # checkpointi geçerse ödül
REWARD_CRASH        = -50 # kaza yaparsa ceza
REWARD_SLOW         = -0.5 # Yavaş gitmesine karşın ceza
REWARD_FORWARD      =  3 # İleri giderse biraz ödül
REWARD_BACKWARD     = -2 # Geri giderse ufak bir ceza 

# ── Q-Learning ─────────────────────────────────────
LEARNING_RATE  = 0.1 # Yeni bilgi gelirse ne kadar ciddi almalı
GAMMA          = 0.95 # Geleceği ne kadar önemseyeyim, 0.95 çünkü gelecek de önemli
EPSILON_START  = 1.0 # Epsilon: rastgele hareket etme olasılığı. Başta rastgele keşif
EPSILON_END    = 0.05 # Epsilon 0.05’in altına düşmesin
EPSILON_DECAY  = 0.998 # Her episode sonunda epsilon düşmeli ki rastgelelik azalsın öğrenme başlasın

# ── Eğitim ─────────────────────────────────────────
MAX_EPISODES   = 2000 # Toplam 2000 oyun 
SAVE_INTERVAL  = 100 # Her 100 episode'da bir modeli kayıt et 

# ── Dosya Yolları ───────────────────────────────────
MODEL_DIR = "models/"
LOG_DIR   = "logs/"