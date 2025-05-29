<template>
  <q-page class="q-pa-md">
    
    <div class="row q-col-gutter-md q-mb-md">
      
      <div class="col-12 flex flex-center q-gutter-sm">
        <div class="text-h5 text-bold q-mr-md">Dıgıdık</div> 
        <q-btn label="Generate" color="primary" @click="generate" />
        <q-btn
          :label="currentRound === -1 ? 'Start' : 'Next'"
          color="secondary"
          :disable="btnDisabled"
          @click="start"
        />
        
        <q-toggle
          v-model="autoActive"
          color="accent"
          checked-icon="play_arrow"
          unchecked-icon="pause"
          label="Auto"
          size="lg"
          :disable="!schedule.length"
        />
      </div>
    </div>
    <transition name="scale" appear>
      <div
        v-if="countDown > 0"
        class="countdown absolute-full flex flex-center"
      >
        {{ countDown }}
      </div>
    </transition>
    
    <div class="row q-col-gutter-md">
      
      <div class="col-3">
        <q-card flat>
          <q-card-section class="text-h6">Scoreboard</q-card-section>
          <q-table dense flat bordered
                  :rows="scoreboard"
                  :columns="scoreCols"
                  row-key="id"
                  :pagination="{rowsPerPage:0}">

            
            <template #body-cell-clr="p">
              <q-td>
                <div :style="{
                      width:'12px',height:'12px',
                      background:p.row.color,border:'1px solid #222'
                    }" />
              </q-td>
            </template>

            <template #body-cell-rank="p"><q-td>{{ p.rowIndex+1 }}</q-td></template>
            <template #body-cell-points="p"><q-td>{{ p.row.points }}</q-td></template>
          </q-table>

          
          <q-separator class="q-mt-sm"/>
          <q-card-section>
            <div>Balance: {{ balance }}</div>

            <q-select v-model="selectedHorse"
                      :options="currentHorses"
                      option-label="name" option-value="id"
                      emit-value map-options dense filled bg-color="white"
                      :disable="raceRunning || !schedule.length" />

            <q-input v-model.number="betAmount"
                     type="number" min="1"
                     dense filled bg-color="white"
                     placeholder="Bet $"
                     class="q-mt-xs"
                     :disable="raceRunning" />

            <q-btn label="Bet"
                   color="primary"
                   class="q-mt-sm full-width"
                   :disable="raceRunning || betLocked || !selectedHorse || betAmount<=0"
                   @click="placeBet" />
          </q-card-section>
        </q-card>
      </div>

     
      <div class="col-6 flex flex-center column items-center">
        <canvas ref="trackCanvas"
                :width="canvasWidth"
                :height="canvasHeight"
                style="border:1px solid #ccc;border-radius:4px;" />
        <div v-if="currentRound >= 0 && currentRound < roundsCount"
            class="text-center q-mt-sm text-h6">
          Round {{ currentRound + 1 }} · {{ distances[currentRound] }} m
        </div>

      </div>

      
      <div class="col-3">
        <q-card flat>
          <q-card-section class="text-h6">Fixture</q-card-section>

          <q-expansion-item-group
            :key="fixtureKey"   
            v-if="schedule.length"
            accordion
            v-model="openRound"
            @update:model-value="handleExp">


            <q-expansion-item v-for="(colorRandom,i) in schedule" :key="i"
                              :label="`Round ${i+1}`" dense>
              <q-list dense bordered>
                <q-item v-for="h in colorRandom" :key="h.id">
                  <q-item-section>
                    <span v-if="h.finishPos">{{ h.finishPos }}.&nbsp;</span>{{ h.name }}
                  </q-item-section>
                </q-item>
              </q-list>
            </q-expansion-item>

          </q-expansion-item-group>

          <div v-else class="text-grey-6 q-pa-sm">
            Press Generate to generate the fixture.
          </div>
        </q-card>
      </div>
    </div>
  </q-page>
</template>

<script setup>
import { ref, reactive, computed, onMounted, nextTick } from 'vue'
import { Notify } from 'quasar'

import horseW1 from 'src/assets/whitehorse1.png'
import horseW2 from 'src/assets/whitehorse2.png'
import horseB1 from 'src/assets/brownhorse1.png'
import horseB2 from 'src/assets/brownhorse2.png'
import horseG1 from 'src/assets/grayhorse1.png'
import horseG2 from 'src/assets/grayhorse2.png'
import jockeySrc from 'src/assets/jockey.png'  

const fixtureKey = ref(0)
const canvasWidth  = 950
const BASE_SCALE   = 4
const BASE_SPEED   = 5
const MIN_DIST     = 1200
const distances    = [1200,1400,1600,1800,2000,2200]
const finishMargin = 80
const horsesPerRound = 10
const totalHorses    = 20
const roundsCount    = 6
const EXTRA_TRACK    = 0
const SADDLE_W_BASE  = 4
const SADDLE_H_BASE  = 4
const SADDLE_Y_BASE  = 15
const JOCKEY_Y_BASE = 0    


const laneHeight   = ref(110)
const canvasHeight = ref(laneHeight.value * horsesPerRound + EXTRA_TRACK)

const trackCanvas = ref(null); let canvas = null
const spritesReady = ref(false)
const framesMap = {}
let jockey = null

async function loadSprites () {
  const imgs = await Promise.all(
    [horseW1,horseW2,horseB1,horseB2,horseG1,horseG2,jockeySrc].map(src => new Promise(r => {
      const img = new Image()
      img.src = src
      img.onload = () => r(img)
    }))
  )
  framesMap.white = [imgs[0], imgs[1]]
  framesMap.brown = [imgs[2], imgs[3]]
  framesMap.gray  = [imgs[4], imgs[5]]
  jockey       = imgs[6]      
  spritesReady.value = true
}

const names = [
  'Şahbatur',
  'Gülbatur',
  'Zıpır Tay',
  'Poyrazşah',
  'Fırtına Doruk',
  'Dörtnala Deli',
  'Pıtırcık',
  'Kıvrak Efe',
  'Çapkın Tay',
  'Şimşek Yiğit',
  'Hopdurak',
  'Kişneme Reis',
  'Gazoz Tay',
  'Külüstür',
  'Gönüldaş',
  'Rüzgâr Faresi',
  'Yeldirme Can',
  'Tornet Doru',
  'Gıdıgıdık',
  'Hınzırbey'
]

const colors = ['white','brown','gray']
const colorRandom = a => a[Math.floor(Math.random()*a.length)]


const balance       = ref(100)
const betAmount     = ref(0)
const selectedHorse = ref(null)
const betHorse      = ref(null)
const betStake      = ref(0)
const betLocked     = computed(() => betHorse.value !== null)
const autoActive = ref(false)       
const countDown  = ref(0)           

const horses = reactive(Array.from({ length: totalHorses }, (_, i) => {
  const baseCond = 80 + Math.floor(Math.random() * 20 - 10)
  return {
    id: i + 1,
    name: names[i],
    spriteKey: colorRandom(colors),
    baseCond,
    currentCond: baseCond,
    color: '#' + Math.floor(Math.random() * 0xffffff).toString(16).padStart(6, '0'),
    fatigueRate: Math.random() * 0.5,
    boost: false,
    x: 0, y: 0, frame: 0, frameTimer: 0, rank: null
  }
}))


const scoreCols = [
  { name:'clr',   label:'',  field:'color', align:'left' },  
  { name:'rank',  label:'#',field:'rank',  align:'left' },
  { name:'name',  label:'Name', field:'name' },
  { name:'points',label:'Pts',  field:'points' }
]


const scoreboard = reactive(
  horses.map(h=>({ id:h.id, name:h.name, color:h.color, points:0 }))
)

const schedule     = ref([])
const currentRound = ref(-1)
const raceRunning  = ref(false)
const openRound    = ref(0)     


const mix = arr => {
  const a = [...arr]
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1))
    ;[a[i], a[j]] = [a[j], a[i]]
  }
  return a
}
const scaleMult = idx => BASE_SCALE * (MIN_DIST / distances[idx])
const speedMult = idx => BASE_SPEED * (MIN_DIST / distances[idx])

const btnDisabled = computed(() =>
  !schedule.value.length || !spritesReady.value ||
  raceRunning.value      || currentRound.value >= roundsCount)

const currentHorses = computed(() => {
  if (!schedule.value.length) return []
  const idx = currentRound.value === -1 ? 0 : currentRound.value
  return schedule.value[idx].map(({ id, name }) => ({ id, name }))
})

let lastOpen = 0
function handleExp (val) {
  if (val === null) {
    openRound.value = lastOpen     
  } else {
    lastOpen = val
  }
}

function placeBet () {
  if (betAmount.value > balance.value) {
    Notify.create({ message: 'Insufficient balance', color: 'negative' })
    return
  }
  betHorse.value = selectedHorse.value
  betStake.value = betAmount.value
  Notify.create({ message: `Bet ${betStake.value}$ on #${betHorse.value}`, color: 'info' })
}

function generate () {
  schedule.value = Array.from({ length: roundsCount }, () =>
    mix(horses).slice(0, horsesPerRound)
      .map(h => ({ id: h.id, name: h.name, finishPos: null }))
  )

  currentRound.value = -1
  raceRunning.value  = false
  countDown.value    = 0
  betHorse.value     = null
  betStake.value     = 0
  selectedHorse.value= null
  autoActive.value   = false 


  scoreboard.forEach(s => { s.points = 0 })
  scoreboard.sort((a,b)=>a.name.localeCompare(b.name))

  
  fixtureKey.value++           

  openRound.value = 0
  lastOpen = 0
}



function start () {
  if (raceRunning.value || currentRound.value >= roundsCount) return

  
  if (currentRound.value === -1) currentRound.value = 0

  
  countDown.value = 3
  const id = setInterval(() => {
    countDown.value--
    if (countDown.value === 0) {
      clearInterval(id)
      round(currentRound.value)
    }
  }, 1000)
}


let RAF = null
function round (idx) {
  if (RAF) cancelAnimationFrame(RAF)
  raceRunning.value = true
  


  const SCALE = scaleMult(idx) *0.8
  const SPEED = speedMult(idx) *0.8

  laneHeight.value   = framesMap.white[0].height * BASE_SCALE * 0.85
  canvasHeight.value = laneHeight.value * horsesPerRound + EXTRA_TRACK

  const pack = schedule.value[idx].map(s => horses.find(h => h.id === s.id))
  let finishCounter = 0
  const finishX = canvasWidth - finishMargin

  pack.forEach((h, i) => {
    h.x = 0
    h.currentCond = h.baseCond
    h.boost = false
    h.rank = null
    h.y = i * laneHeight.value + (laneHeight.value - framesMap[h.spriteKey][0].height * SCALE) / 2
    h.finished = false
    h.order    = 11
  })

  function step () {
    canvas.clearRect(0, 0, canvasWidth, canvasHeight.value)
    canvas.fillStyle = '#2ecc71'
    canvas.fillRect(0, 0, canvasWidth, canvasHeight.value)
    canvas.strokeStyle = '#ffffff55'
    canvas.lineWidth = 1
    for (let y = laneHeight.value; y < canvasHeight.value; y += laneHeight.value) {
      canvas.beginPath(); canvas.moveTo(0, y); canvas.lineTo(canvasWidth, y); canvas.stroke()
    }
    canvas.strokeStyle = '#f00'
    canvas.lineWidth = 3
    canvas.beginPath(); canvas.moveTo(finishX, 0); canvas.lineTo(finishX, canvasHeight.value); canvas.stroke()

   pack.forEach(h => {
      const done = h.x >= finishX
      if (done) {
        
        if (!h.finished) {
          h.finished = true;
          h.order    = ++finishCounter;
        }
        
        h.x = finishX;
      } else {
        
        if (h.x > finishX * 0.4) {
          h.currentCond = Math.max(h.currentCond - h.fatigueRate, 50);
        }

        if (!h.boost && h.x > finishX * 0.7) {
          h.currentCond += 10;
          h.boost = true;
        }

        h.x += (h.currentCond / 100) * SPEED;

        
        if (++h.frameTimer > 6) {
          h.frame = 1 - h.frame;
          h.frameTimer = 0;
        }
      }


      const img = framesMap[h.spriteKey][h.frame]
      const w   = img.width  * SCALE
      const hh  = img.height * SCALE

    
      canvas.drawImage(img, h.x, h.y, w, hh)

      
      canvas.fillStyle = h.color
      canvas.fillRect(
        h.x + (w - SADDLE_W_BASE * SCALE) / 2.8,
        h.y + SADDLE_Y_BASE * SCALE,
        SADDLE_W_BASE * SCALE,
        SADDLE_H_BASE * SCALE
      )

      
      const jw = jockey.width  * SCALE* 0.98    
      const jh = jockey.height * SCALE* 0.98
      canvas.drawImage(
        jockey,
        h.x + (w - jw) / 2,
        h.y + (JOCKEY_Y_BASE * SCALE),
        jw, jh
      )
    })

    pack.slice().sort((a,b)=>a.order-b.order).forEach((h,i)=>h.rank=i+1)
    if (finishCounter === pack.length) finish(pack, idx)
    else RAF = requestAnimationFrame(step)
  }
  RAF = requestAnimationFrame(step)
}

  

function finish (pack, idx) {
  const sorted = pack.slice().sort((a,b)=>a.order-b.order)
  const pts = [10, 5, 4, 3, 2, 1]
  sorted.forEach((h, i) => {
    const row = scoreboard.find(s => s.id === h.id)
    if (row && i < pts.length) row.points += pts[i]
  })
  scoreboard.sort((a, b) => b.points - a.points || a.name.localeCompare(b.name))

  schedule.value[idx] = sorted.map((h, i) => ({
    id: h.id, name: h.name, finishPos: i + 1
  }))

  
  if (betHorse.value != null) {
    const winner = sorted[0].id
    if (winner === betHorse.value) {
      balance.value += betStake.value * 9
      Notify.create({ message: `WIN! +${betStake.value * 9}$`, color: 'positive' })
    } else {
      balance.value -= betStake.value
      Notify.create({ message: `Lost ${betStake.value}$`, color: 'negative' })
    }
  }
  betHorse.value = null
  betStake.value = 0
  selectedHorse.value = null

   raceRunning.value = false
    currentRound.value++       

    if (currentRound.value >= roundsCount) {
      
      currentRound.value = -1        

     
      if (autoActive.value) {
        setTimeout(() => {
          generate()  
          start()     
        }, 6000)     
      }

      return  
    }

    nextTick(() => {})
    if (autoActive.value) {
      setTimeout(() => start(), 1000)
    }
  }



onMounted(async () => {
  canvas = trackCanvas.value.getContext('2d')
  await loadSprites()
})
</script>

<style>
@import url('https://fonts.googleapis.com/css2?family=Press+Start+2P&display=swap');
.q-page,body{
  background:#63c2e5;
  color:#111;
  font-family:'Press Start 2P',monospace;font-size:12px;
  line-height:1.4;}
.q-card,.q-table,canvas{border:1px solid #3a3f54!important;border-radius:4px;}
.q-btn__content{font-family:'Press Start 2P',monospace;}
.q-expansion-item__container,.q-table th{font-family:'Press Start 2P',monospace;font-size:11px;}
.q-table td{color:#111;}
canvas{image-rendering:pixelated;}
.q-notification{font-family:'Press Start 2P',monospace;color:#111;}
.countdown{
  font-size:120px;
  font-weight:bold;
  color:#fff;
  text-shadow:0 0 10px #000;
  pointer-events:none;
}
.scale-enter-active,.scale-leave-active{
  transition:transform .3s cubic-bezier(.4,0,.2,1),opacity .3s;
}
.scale-enter-from,.scale-leave-to{
  transform:scale(.2);
  opacity:0;
}
</style>
