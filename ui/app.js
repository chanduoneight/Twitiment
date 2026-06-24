const API_BASE = 'http://127.0.0.1:8000/api';

document.addEventListener('DOMContentLoaded', () => {
    // Tab switching logic
    const navItems = document.querySelectorAll('.nav-item');
    const tabPanes = document.querySelectorAll('.tab-pane');

    navItems.forEach(item => {
        item.addEventListener('click', () => {
            // Remove active class
            navItems.forEach(n => n.classList.remove('active'));
            tabPanes.forEach(t => t.classList.remove('active'));
            
            // Add active class
            item.classList.add('active');
            const targetId = item.getAttribute('data-tab');
            document.getElementById(targetId).classList.add('active');
        });
    });

    // Text Analyzer
    const analyzeTextBtn = document.getElementById('analyze-text-btn');
    analyzeTextBtn.addEventListener('click', async () => {
        const text = document.getElementById('text-input').value;
        if (!text) return;
        
        await handleApiCall(
            () => fetch(`${API_BASE}/analyze`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text })
            }),
            'text-results',
            (data, container) => {
                container.innerHTML = ''; // Clear
                container.appendChild(renderSentimentResult(data));
            }
        );
    });

    // Tweet Analyzer
    const analyzeTweetBtn = document.getElementById('analyze-tweet-btn');
    analyzeTweetBtn.addEventListener('click', async () => {
        const url = document.getElementById('tweet-url-input').value;
        if (!url) return;
        
        await handleApiCall(
            () => fetch(`${API_BASE}/tweet?url=${encodeURIComponent(url)}`),
            'tweet-results',
            (data, container) => {
                container.innerHTML = '';
                container.appendChild(renderTweetCard(data.tweet));
                container.appendChild(renderSentimentResult(data));
            }
        );
    });

    // Hashtag Search
    const searchHashtagBtn = document.getElementById('search-hashtag-btn');
    searchHashtagBtn.addEventListener('click', async () => {
        const hashtag = document.getElementById('hashtag-input').value;
        if (!hashtag) return;
        
        await handleApiCall(
            () => fetch(`${API_BASE}/hashtag?hashtag=${encodeURIComponent(hashtag)}`),
            'hashtag-results',
            (data, container) => {
                container.innerHTML = '';
                if (data.tweets.length === 0) {
                    container.innerHTML = '<p>No tweets found.</p>';
                    return;
                }
                data.tweets.forEach(item => {
                    const wrap = document.createElement('div');
                    wrap.style.marginBottom = '4rem';
                    wrap.appendChild(renderTweetCard(item.tweet));
                    wrap.appendChild(renderSentimentResult(item));
                    container.appendChild(wrap);
                });
            }
        );
    });
});

async function handleApiCall(apiFn, resultsContainerId, renderFn) {
    const loader = document.getElementById('loader');
    const resultsContainer = document.getElementById(resultsContainerId);
    
    loader.classList.remove('hidden');
    resultsContainer.classList.add('hidden');
    
    try {
        const response = await apiFn();
        if (!response.ok) {
            const err = await response.json();
            throw new Error(err.detail || 'API Error');
        }
        const data = await response.json();
        
        renderFn(data, resultsContainer);
        resultsContainer.classList.remove('hidden');
    } catch (e) {
        alert('Error: ' + e.message);
    } finally {
        loader.classList.add('hidden');
    }
}

function renderSentimentResult(data) {
    const template = document.getElementById('sentiment-result-template').content.cloneNode(true);
    
    const sent = data.sentence_sentiment || data.sentiment;
    const words = data.word_sentiment || [];
    const metrics = data.metrics || null;
    
    // Overall score — prefer metrics.mean_compound when available
    const scoreVal = template.querySelector('.score-value');
    const scoreLabel = template.querySelector('.score-label');
    
    if (metrics) {
        scoreVal.textContent = metrics.mean_compound.toFixed(2);
        scoreLabel.textContent = metrics.category;
    } else {
        scoreVal.textContent = sent.compound.toFixed(2);
        scoreLabel.textContent = sent.label;
    }
    
    const displayCompound = metrics ? metrics.mean_compound : sent.compound;
    let color = 'var(--text-secondary)';
    if (displayCompound >= 0.05) color = 'var(--retweet-green)';
    else if (displayCompound <= -0.05) color = 'var(--like-pink)';
    
    scoreVal.style.color = color;

    // Breakdown — prefer metrics.distribution (token ratio) when available
    if (metrics) {
        const dist = metrics.distribution;
        template.querySelector('.pos-val').textContent = dist.positive.toFixed(1) + '%';
        template.querySelector('.neu-val').textContent = dist.neutral.toFixed(1) + '%';
        template.querySelector('.neg-val').textContent = dist.negative.toFixed(1) + '%';
        setTimeout(() => {
            template.querySelector('.pos-fill').style.width = dist.positive + '%';
            template.querySelector('.neu-fill').style.width = dist.neutral + '%';
            template.querySelector('.neg-fill').style.width = dist.negative + '%';
        }, 100);
    } else {
        template.querySelector('.pos-val').textContent = (sent.pos * 100).toFixed(1) + '%';
        template.querySelector('.neu-val').textContent = (sent.neu * 100).toFixed(1) + '%';
        template.querySelector('.neg-val').textContent = (sent.neg * 100).toFixed(1) + '%';
        setTimeout(() => {
            template.querySelector('.pos-fill').style.width = (sent.pos * 100) + '%';
            template.querySelector('.neu-fill').style.width = (sent.neu * 100) + '%';
            template.querySelector('.neg-fill').style.width = (sent.neg * 100) + '%';
        }, 100);
    }

    // Words
    const wordChips = template.querySelector('.word-chips');
    if (words && words.length > 0) {
        words.forEach((w, i) => {
            const chip = document.createElement('div');
            chip.className = `chip ${w.label}`;
            chip.textContent = `${w.word} (${w.compound.toFixed(2)})`;
            chip.style.animationDelay = `${i * 0.05}s`;
            wordChips.appendChild(chip);
        });
    } else {
        template.querySelector('.word-card').style.display = 'none';
    }

    // Pipeline
    if (data.pipeline_steps && data.pipeline_steps.length > 0) {
        const prepCard = template.querySelector('.preprocessing-card');
        prepCard.style.display = 'block';
        const stepsContainer = prepCard.querySelector('.pipeline-steps');
        data.pipeline_steps.forEach(step => {
            const stepDiv = document.createElement('div');
            stepDiv.className = 'pipeline-step';
            stepDiv.innerHTML = `
                <div class="step-header">
                    <span class="step-number">${step.step_no}</span>
                    <div class="step-title">
                        <strong>${step.step_name}</strong>
                        <span class="step-desc">${step.description}</span>
                    </div>
                </div>
                <div class="step-result">${step.result_text}</div>
            `;
            stepsContainer.appendChild(stepDiv);
        });
    }

    return template;
}

function renderTweetCard(tweet) {
    const template = document.getElementById('tweet-card-template').content.cloneNode(true);
    
    template.querySelector('.author-id').textContent = `@${tweet.author_id}`;
    template.querySelector('.tweet-date').textContent = new Date(tweet.created_at).toLocaleString();
    template.querySelector('.tweet-text').textContent = tweet.text;
    
    template.querySelector('.metric-rt').textContent = tweet.public_metrics.retweet_count;
    template.querySelector('.metric-like').textContent = tweet.public_metrics.like_count;
    template.querySelector('.metric-reply').textContent = tweet.public_metrics.reply_count;
    
    return template;
}
